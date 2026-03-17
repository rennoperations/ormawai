import os
import streamlit as st
from pathlib import Path
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import docx

# ─────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────
GROQ_MODEL    = "llama-3.1-8b-instant"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3
COLLECTION    = "ormawai"
DOCS_DIR      = Path("docs")


# ─────────────────────────────────────────
#  INISIALISASI
# ─────────────────────────────────────────
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="vectorstore")
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


@st.cache_resource
def init_groq():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("GROQ_API_KEY belum diset. Tambahkan di Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)


# ─────────────────────────────────────────
#  PEMROSESAN DOKUMEN
# ─────────────────────────────────────────
def extract_text(file_path: Path) -> str:
    text   = ""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

    elif suffix in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"

    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")

    return text.strip()


def chunk_text(text: str, source_name: str) -> list[dict]:
    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({
                "id":     f"{source_name}__chunk_{idx}",
                "text":   chunk,
                "source": source_name
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_documents(collection) -> int:
    supported = {".pdf", ".docx", ".doc", ".txt"}
    files     = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in supported]
    indexed   = 0

    for file_path in files:
        source_name = file_path.stem
        existing    = collection.get(where={"source": source_name})
        if existing["ids"]:
            continue

        text   = extract_text(file_path)
        chunks = chunk_text(text, source_name)
        if not chunks:
            continue

        collection.add(
            ids       = [c["id"]   for c in chunks],
            documents = [c["text"] for c in chunks],
            metadatas = [{"source": c["source"]} for c in chunks]
        )
        indexed += len(chunks)

    return indexed


# ─────────────────────────────────────────
#  RAG PIPELINE
# ─────────────────────────────────────────
def retrieve(collection, query: str) -> tuple[str, list[str]]:
    results = collection.query(query_texts=[query], n_results=TOP_K)
    chunks  = results["documents"][0]
    metas   = results["metadatas"][0]
    sources = sorted({m["source"] for m in metas})
    context = "\n\n---\n\n".join(chunks)
    return context, sources


def generate_answer(client: Groq, query: str, context: str) -> str:
    system_prompt = """Kamu adalah OrmawAI, asisten resmi untuk Badan Eksekutif Mahasiswa (BEM) dan organisasi kemahasiswaan.

Tugasmu adalah membantu pengurus BEM menemukan informasi dari dokumen resmi organisasi seperti AD/ART, SOP kegiatan, format proposal, dan rundown acara.

Aturan yang WAJIB diikuti:
1. Jawab HANYA berdasarkan konteks dokumen yang diberikan.
2. Jika informasi tidak ada di dokumen, katakan: "Informasi ini tidak tersedia dalam dokumen yang saya miliki saat ini."
3. Selalu sebutkan nama dokumen sumber informasi tersebut berasal.
4. Gunakan bahasa Indonesia yang formal namun mudah dipahami.
5. Jika pertanyaan tentang SOP atau prosedur, tampilkan dalam format langkah bernomor.
6. Jangan menambahkan informasi dari luar dokumen yang diberikan."""

    user_prompt = f"""Pertanyaan: {query}

Konteks dari dokumen organisasi:
{context}

Jawablah pertanyaan di atas berdasarkan konteks yang diberikan."""

    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature = 0.1,
        max_tokens  = 1024
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────
def main():
    st.set_page_config(page_title="OrmawAI", page_icon="🎓", layout="wide")

    st.title("🎓 OrmawAI")
    st.caption("Asisten cerdas berbasis dokumen resmi BEM & Organisasi Kemahasiswaan")
    st.divider()

    collection  = init_chromadb()
    groq_client = init_groq()

    # ── Sidebar ──────────────────────────
    with st.sidebar:
        st.header("📁 Manajemen Dokumen")
        st.caption("Upload dokumen BEM: AD/ART, SOP, format proposal, rundown, dll.")

        uploaded_files = st.file_uploader(
            label                 = "Upload dokumen",
            type                  = ["pdf", "docx", "txt"],
            accept_multiple_files = True
        )

        if uploaded_files:
            DOCS_DIR.mkdir(exist_ok=True)
            for uf in uploaded_files:
                (DOCS_DIR / uf.name).write_bytes(uf.read())

            with st.spinner("Mengindeks dokumen..."):
                n = index_documents(collection)

            if n > 0:
                st.success(f"✅ {n} chunk berhasil diindeks!")
            else:
                st.info("ℹ️  Semua dokumen sudah diindeks sebelumnya.")

        st.divider()
        st.subheader("📋 Dokumen Terindeks")
        all_docs = collection.get()
        if all_docs["ids"]:
            sources = sorted({m["source"] for m in all_docs["metadatas"]})
            for src in sources:
                st.markdown(f"- 📄 `{src}`")
            st.caption(f"Total: {len(all_docs['ids'])} chunk")
        else:
            st.info("Belum ada dokumen. Upload terlebih dahulu.")

        st.divider()
        if st.button("🗑️ Reset semua dokumen", type="secondary", use_container_width=True):
            collection.delete(where={"source": {"$ne": "___nonexistent___"}})
            st.cache_resource.clear()
            st.rerun()

    # ── Main area ─────────────────────────
    col_chat, col_history = st.columns([2, 1])

    with col_chat:
        st.subheader("💬 Tanya OrmawAI")

        # Tombol contoh pertanyaan
        st.caption("Contoh pertanyaan cepat:")
        ex_cols = st.columns(3)
        examples = [
            "Apa syarat mendirikan kepanitiaan?",
            "Bagaimana format proposal kegiatan?",
            "Apa struktur organisasi BEM?"
        ]
        for i, ex in enumerate(examples):
            with ex_cols[i]:
                if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                    st.session_state["prefill"] = ex

        query = st.text_area(
            label       = "Pertanyaan:",
            value       = st.session_state.get("prefill", ""),
            height      = 100,
            placeholder = "Contoh: Bagaimana prosedur pengajuan proposal kegiatan BEM?"
        )

        if st.button("🔍 Cari Jawaban", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Masukkan pertanyaan terlebih dahulu.")
            elif not collection.get()["ids"]:
                st.warning("⚠️  Belum ada dokumen. Upload dokumen BEM di sidebar kiri.")
            else:
                with st.spinner("Mencari di dokumen..."):
                    context, sources = retrieve(collection, query)
                with st.spinner("Menyusun jawaban..."):
                    answer = generate_answer(groq_client, query, context)

                st.divider()
                st.markdown("### 📝 Jawaban")
                st.markdown(answer)
                st.divider()
                st.markdown("**📚 Sumber dokumen:**")
                for src in sources:
                    st.markdown(f"- `{src}`")

                # simpan ke riwayat
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].insert(0, {
                    "q": query, "a": answer, "sources": sources
                })
                # reset prefill
                st.session_state.pop("prefill", None)

    with col_history:
        st.subheader("🕐 Riwayat")
        history = st.session_state.get("history", [])
        if history:
            for i, item in enumerate(history[:5]):
                with st.expander(f"Q: {item['q'][:45]}...", expanded=(i == 0)):
                    st.markdown(item["a"])
                    st.caption(f"Sumber: {', '.join(item['sources'])}")
        else:
            st.info("Belum ada riwayat pertanyaan.")


if __name__ == "__main__":
    main()