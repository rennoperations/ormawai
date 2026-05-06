import os
import re
import shutil
import streamlit as st
from pathlib import Path
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import docx
import pandas as pd
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────
GROQ_MODEL      = "llama-3.1-8b-instant"
CHUNK_SIZE      = 800    # lebih besar agar konteks tidak terpotong
CHUNK_OVERLAP   = 150    # overlap lebih besar agar antar-chunk saling nyambung
TOP_K_RETRIEVE  = 10     # ambil lebih banyak kandidat dulu
TOP_K_RERANK    = 5      # setelah reranker, ambil 5 terbaik
COLLECTION      = "ormawai"
DOCS_DIR        = Path("docs")
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────
#  SVG ICONS (line art, thin stroke)
# ─────────────────────────────────────────
def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    s = size
    c = color
    icons = {
        "graduation": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>',
        "upload":     f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "file":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
        "search":     f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "clock":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "trash":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
        "menu":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>',
        "close":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
        "check":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "book":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "layers":     f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "info":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "chevron-right": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>',
        "sidebar":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>',
    }
    return icons.get(name, "")


# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Font Claude — system-ui fallback ke inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* Background netral seperti Claude */
.stApp { background: #1a1a1a !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    min-width: 268px !important;
    width: 268px !important;
}
[data-testid="stSidebarContent"] { padding: 18px 14px !important; }
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.55) !important; font-size: 0.82rem !important; }

/* Hero — clean, minimal */
.hero {
    background: #222222;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero-top { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.hero-icon {
    width: 40px; height: 40px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.hero h1 {
    font-size: 1.5rem; font-weight: 600;
    color: #ececec; margin: 0;
    letter-spacing: -0.3px;
}
.hero-sub { color: rgba(255,255,255,0.4); font-size: 0.875rem; line-height: 1.6; margin: 0; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.45);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 500; margin-bottom: 12px;
}

/* Section labels */
.sec-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: rgba(255,255,255,0.25); margin-bottom: 8px;
    display: flex; align-items: center; gap: 5px;
}

/* Primary button — sama dengan Claude */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #ffffff !important;
    border: none !important;
    color: #111111 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 12px !important;
    border-radius: 8px !important;
    letter-spacing: 0.1px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #e8e8e8 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Textarea */
textarea {
    background: #2a2a2a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus {
    border-color: rgba(255,255,255,0.25) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Answer box — seperti bubble Claude */
.answer-box {
    background: #2a2a2a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 20px;
    line-height: 1.8;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.82);
    margin: 10px 0 14px;
    font-family: 'Inter', sans-serif;
}
.answer-box p { margin: 0 0 10px 0; }
.answer-box p:last-child { margin-bottom: 0; }
.answer-box ol, .answer-box ul { padding-left: 20px; margin: 6px 0; }
.answer-box li { margin-bottom: 4px; }

/* Source badge */
.src-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.5);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.75rem; margin: 3px 4px 3px 0;
}

/* Stat chip sidebar */
.stat-chip {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 7px 10px;
    margin: 4px 0; font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
}
.dot-green { width:6px; height:6px; border-radius:50%; background:#4ade80; flex-shrink:0; }

/* History card */
.hist-card {
    background: #222222;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.hist-card:hover { border-color: rgba(255,255,255,0.15); }
.hist-q { font-size: 0.78rem; color: rgba(255,255,255,0.4); margin-bottom: 5px; }
.hist-a { font-size: 0.84rem; color: rgba(255,255,255,0.7); line-height: 1.55; }
.hist-src { margin-top:6px; font-size:0.7rem; color:rgba(255,255,255,0.22); }

/* File uploader */
[data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.02) !important;
    transition: all 0.15s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(255,255,255,0.25) !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Drag & drop dokumen di sini";
    display: block; font-size: 0.82rem; font-weight: 500;
    color: rgba(255,255,255,0.5); text-align: center; margin-bottom: 4px;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display: none !important; }

/* Secondary button (Reset) */
div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.8rem !important;
    border-radius: 7px !important;
    padding: 8px !important;
    transition: all 0.15s !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: rgba(255,255,255,0.25) !important;
    color: rgba(255,255,255,0.75) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div { background: rgba(255,255,255,0.15) !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div > div { background: #ffffff !important; border-radius: 4px !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 14px 0 !important; }

/* Sidebar force expanded */
[data-testid="stSidebar"][aria-expanded="false"] {
    display: flex !important; transform: none !important; min-width: 268px !important;
}
</style>
"""


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


def hard_reset_vectorstore():
    """
    Reset total yang aman di Windows.
    Tidak menghapus folder (Windows mengunci file .bin ChromaDB),
    tapi menghapus semua data dari dalam collection dan menghapus docs.
    """
    # Step 1: Ambil client yang sedang aktif dari cache Streamlit
    # lalu hapus dan buat ulang collection (ini aman karena tidak menyentuh file)
    try:
        client = chromadb.PersistentClient(path="vectorstore")
        # Hapus collection lama dan buat baru yang kosong
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        # Buat collection kosong baru
        emb_fn = embedding_functions.DefaultEmbeddingFunction()
        client.create_collection(
            name=COLLECTION,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        pass

    # Step 2: Hapus folder docs agar file lama tidak ter-index ulang
    if DOCS_DIR.exists():
        try:
            shutil.rmtree(DOCS_DIR)
        except Exception:
            # Fallback: hapus file satu per satu
            for f in DOCS_DIR.rglob("*"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

    # Step 3: Clear cache Streamlit agar init_chromadb() dipanggil ulang
    st.cache_resource.clear()


@st.cache_resource
def init_groq():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("GROQ_API_KEY belum diset. Tambahkan di Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource
def init_reranker():
    """
    Load CrossEncoder reranker.
    Model ~80MB, download otomatis saat pertama kali dijalankan.
    Cached agar tidak reload setiap query.
    """
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        st.warning(f"Reranker tidak tersedia ({e}). Menggunakan retrieval standar.")
        return None


# ─────────────────────────────────────────
#  PEMROSESAN DOKUMEN
# ─────────────────────────────────────────
def is_valid(val) -> bool:
    """Cek apakah nilai cell tidak kosong/NaN."""
    return str(val).strip().lower() not in ("nan", "none", "", "nat")


def find_header_row(df) -> int:
    """
    Deteksi baris header asli dalam Excel yang headernya tidak di baris 0.
    Heuristik: baris header biasanya punya banyak nilai string pendek dan unik.
    """
    for i, row in df.iterrows():
        vals = [str(v).strip() for v in row if is_valid(v)]
        if (len(vals) >= 2
                and all(len(v) < 60 for v in vals)
                and len(vals) == len(set(vals))):  # semua nilai unik = kandidat header
            return i
    return -1


# Kata kunci yang menandai baris header tabel (case-insensitive)
HEADER_KEYWORDS = {
    "no", "no.", "nama", "name", "tanggal", "date", "waktu", "time",
    "pic", "penanggung jawab", "kategori", "category", "program", "type",
    "tipe", "jenis", "schedule", "jadwal", "keterangan", "notes", "status",
    "kegiatan", "activity", "proker", "jabatan", "position", "divisi",
    "division", "link", "deadline", "format", "aturan", "rule",
    "project category", "program name", "program type"
}


def is_header_row(row) -> bool:
    """
    Deteksi apakah baris ini adalah header tabel.
    Header harus: semua nilai string pendek, unik, dan minimal satu cocok dengan kata kunci.
    """
    vals = [str(v).strip() for v in row if is_valid(v)]
    if len(vals) < 2:
        return False
    # Semua nilai harus string pendek
    if not all(len(v) < 60 for v in vals):
        return False
    # Semua nilai harus unik
    if len(vals) != len(set(v.lower() for v in vals)):
        return False
    # Minimal satu nilai harus cocok keyword header
    vals_lower = {v.lower() for v in vals}
    # Exact match saja — hindari false positive seperti "One-time" cocok "time"
    if not any(kw in vals_lower for kw in HEADER_KEYWORDS):
        return False
    return True


def extract_blocks_from_sheet(df, source_name: str) -> list[str]:
    """
    Ekstrak teks dari sheet Excel dengan pendekatan dua jalur:
    1. Teks panjang (paragraf) → simpan langsung sebagai blok
    2. Sub-tabel → deteksi header dengan keyword matching, konversi tiap baris ke natural language
    """
    blocks = []
    df_raw = df.copy()
    n = len(df_raw)

    # Jalur 1: teks panjang (paragraf instruksi, keterangan)
    for _, row in df_raw.iterrows():
        for val in row:
            v = str(val).strip()
            if len(v) > 80 and v.lower() not in ("nan", "none"):
                blocks.append(v)

    # Jalur 2: deteksi sub-tabel berdasarkan header keyword
    i = 0
    while i < n:
        row = df_raw.iloc[i]
        if is_header_row(row):
            # Ini adalah header — ambil nama kolom
            col_names = [str(v).strip() if is_valid(v) else None
                         for v in row]

            # Ambil baris data setelah header — toleransi 2 baris kosong berturut-turut
            j = i + 1
            empty_streak = 0
            while j < n:
                data_row = df_raw.iloc[j]
                data_vals = [v for v in data_row if is_valid(v)]

                if not data_vals:
                    empty_streak += 1
                    if empty_streak >= 2:  # 2 baris kosong berturut = akhir tabel
                        break
                    j += 1
                    continue

                empty_streak = 0

                if is_header_row(data_row):  # ketemu header tabel baru
                    break

                # Bentuk representasi natural language dari baris ini
                parts = []
                row_vals = list(data_row)
                for k, col_name in enumerate(col_names):
                    if col_name and k < len(row_vals):
                        v = str(row_vals[k]).strip()
                        if is_valid(v):
                            parts.append(f"{col_name}: {v}")

                if parts:
                    blocks.append(f"[{source_name}] " + " | ".join(parts))
                j += 1
            i = j
        else:
            i += 1

    return blocks


def dataframe_to_text(df, source_name: str) -> str:
    """Ubah DataFrame ke teks yang kaya konteks untuk RAG."""
    blocks = extract_blocks_from_sheet(df, source_name)
    if not blocks:
        # Fallback: dump semua nilai non-null
        lines = [f"Dokumen: {source_name}"]
        for _, row in df.iterrows():
            parts = [str(v).strip() for v in row if is_valid(v)]
            if parts:
                lines.append(" | ".join(parts))
        return "\n".join(lines)
    return f"Dokumen: {source_name}\n\n" + "\n".join(blocks)




def extract_calendar_text(df: "pd.DataFrame", file_stem: str) -> str:
    """
    Ekstrak kalender grid (Mon-Sun format) menjadi teks per-tanggal.
    Format asli: baris angka tanggal → baris-baris kegiatan di bawahnya.
    Output: "JANUARI 2026, Senin 5: Perwalian I | Isi KRS A"
    """
    lines = [f"=== {file_stem} — Kalender Kegiatan ===\n"]
    current_month = ""
    # 7 kolom pertama = hari (MON-SUN), kolom ke-7 ke atas = highlight
    DAY_COLS = 7

    i = 0
    n = len(df)
    day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

    while i < n:
        row = df.iloc[i]
        row_vals = [str(v).strip() for v in row]

        # Deteksi header bulan (1 nilai, format "BULAN TAHUN")
        non_empty = [v for v in row_vals if v and v.lower() not in ("nan","none","")]
        if len(non_empty) == 1 and any(m in non_empty[0].upper() for m in
            ["JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY",
             "AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
             "JANUARI","FEBRUARI","MARET","APRIL","MEI","JUNI","JULI",
             "AGUSTUS","SEPTEMBER","OKTOBER","NOVEMBER","DESEMBER"]):
            current_month = non_empty[0]
            i += 1
            continue

        # Deteksi baris tanggal: berisi angka 1-31 di posisi kolom hari
        date_cells = {}
        for col_idx in range(min(DAY_COLS, len(row_vals))):
            v = row_vals[col_idx]
            if v and v.isdigit() and 1 <= int(v) <= 31:
                date_cells[col_idx] = int(v)

        if len(date_cells) >= 1:
            # Kumpulkan kegiatan dari baris-baris berikutnya sampai baris tanggal berikutnya
            event_rows = []
            j = i + 1
            while j < n:
                next_row = df.iloc[j]
                next_vals = [str(v).strip() for v in next_row]
                next_non_empty = [v for v in next_vals if v and v.lower() not in ("nan","none","")]

                # Baris kosong atau baris bulan baru → stop
                if not next_non_empty:
                    j += 1
                    continue

                # Cek apakah ini baris tanggal berikutnya
                next_dates = [v for v in next_vals[:DAY_COLS] if v and v.isdigit() and 1 <= int(v) <= 31]
                if len(next_dates) >= 1:
                    break

                # Cek apakah ini header bulan baru
                if len(next_non_empty) == 1 and any(m in next_non_empty[0].upper() for m in
                    ["JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY","AUGUST",
                     "SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER"]):
                    break

                event_rows.append(next_vals[:DAY_COLS])
                j += 1

            # Untuk setiap tanggal di baris ini, kumpulkan kegiatannya
            for col_idx, date_num in date_cells.items():
                events = []
                for event_row in event_rows:
                    if col_idx < len(event_row):
                        ev = event_row[col_idx]
                        if ev and ev.lower() not in ("nan", "none", ""):
                            events.append(ev)

                day_name = day_names[col_idx] if col_idx < len(day_names) else ""
                date_str = f"{current_month}, {day_name} tanggal {date_num}"
                if events:
                    lines.append(f"{date_str}: {' | '.join(events)}")
                else:
                    lines.append(f"{date_str}: (tidak ada kegiatan)")

            i = j
            continue

        i += 1

    return "\n".join(lines)

def excel_sheet_to_text(df: "pd.DataFrame", sheet_name: str, file_stem: str) -> str:
    """
    Konversi sheet Excel ke teks kaya konteks untuk RAG.
    Mendeteksi blok anggota (data horizontal per-kolom) dan mengubahnya
    menjadi profil per-orang yang mudah dicari AI.
    """
    source = f"{file_stem} — Sheet: {sheet_name}"
    lines = [f"=== {source} ===\n"]
    df = df.copy()

    # --- Deteksi blok anggota horizontal ---
    # Pola: baris nickname pendek (AURA, REZA, dll) diikuti baris data (nama, NIM, jabatan, dll)
    # Kita cari pola ini dan transpose menjadi profil per-orang

    # Label baris yang biasanya ada di blok anggota
    member_row_labels = [
        "nama lengkap", "nim", "jabatan", "ttl", "tempat", "tanggal lahir",
        "email", "instagram", "whatsapp", "wa", "phone", "no hp",
        "ketua", "sekretaris", "bendahara", "staff", "nama", "nickname"
    ]

    n_rows = len(df)
    n_cols = len(df.columns)
    processed_rows = set()

    # Cari blok anggota: baris nickname → baris-baris data → baris kosong
    i = 0
    while i < n_rows:
        row = df.iloc[i]
        row_vals = [str(v).strip() for v in row]
        non_empty = [v for v in row_vals if v and v.lower() not in ("nan","none","")]

        # Deteksi baris nickname: semua nilai pendek (<= 10 char), kapital, >= 3 kolom
        if (len(non_empty) >= 3
                and all(len(v) <= 15 and v == v.upper() and v.isalpha() for v in non_empty)):

            # Ini kemungkinan baris nickname — kumpulkan baris berikutnya sebagai atribut
            nicknames = non_empty
            col_indices = [j for j, v in enumerate(row_vals)
                          if v and v.lower() not in ("nan","none","") and len(v) <= 15 and v == v.upper() and v.isalpha()]

            member_data = {nick: {} for nick in nicknames}
            attr_rows = []

            j = i + 1
            while j < n_rows:
                next_row = df.iloc[j]
                next_vals = [str(v).strip() for v in next_row]
                next_non_empty = [v for v in next_vals if v and v.lower() not in ("nan","none","")]

                if not next_non_empty:  # baris kosong, akhir blok
                    break

                # Cek apakah ini baris nickname baru
                if (len(next_non_empty) >= 3
                        and all(len(v) <= 15 and v == v.upper() and v.isalpha() for v in next_non_empty)):
                    # Blok anggota baru dimulai
                    break

                attr_rows.append(next_vals)
                processed_rows.add(j)
                j += 1

            # Tentukan label atribut dari urutan baris
            attr_labels = ["Nama Lengkap", "NIM", "Jabatan", "Tempat Lahir",
                          "Email", "Instagram", "WhatsApp",
                          "Nama Lengkap 2", "NIM 2", "Jabatan 2", "TTL 2",
                          "Email 2", "Instagram 2", "WhatsApp 2"]

            # Transpose: untuk setiap nickname, kumpulkan nilainya dari tiap baris atribut
            for nick_idx, (col_idx, nick) in enumerate(zip(col_indices, nicknames)):
                profile_parts = [f"Nickname: {nick}"]
                for attr_idx, attr_row in enumerate(attr_rows):
                    if col_idx < len(attr_row):
                        val = attr_row[col_idx]
                        if val and val.lower() not in ("nan", "none", ""):
                            label = attr_labels[attr_idx] if attr_idx < len(attr_labels) else f"Atribut_{attr_idx}"
                            profile_parts.append(f"{label}: {val}")
                if len(profile_parts) > 1:
                    lines.append("Anggota — " + " | ".join(profile_parts))

            processed_rows.add(i)
            i = j
            continue

        i += 1

    # --- Proses baris yang belum diproses (tabel normal, kalender, teks) ---
    current_header = []
    for i, row in df.iterrows():
        if i in processed_rows:
            continue

        row_vals = [str(v).strip() for v in row]
        non_empty = [v for v in row_vals if v and v.lower() not in ("nan","none","")]

        if not non_empty:
            current_header = []
            continue

        # Teks panjang (instruksi, keterangan)
        long_texts = [v for v in non_empty if len(v) > 80]
        if long_texts:
            for t in long_texts:
                lines.append(t)
            continue

        # Deteksi header tabel
        vals_lower = {v.lower() for v in non_empty}
        header_kws = {"no","no.","nama","name","tanggal","date","pic","program",
                      "schedule","kegiatan","jabatan","link","status","monday",
                      "tuesday","wednesday","thursday","friday","saturday","sunday",
                      "program kerja","program name","project category"}
        is_tbl_header = (len(non_empty) >= 2
                         and len(non_empty) == len(set(v.lower() for v in non_empty))
                         and all(len(v) < 50 for v in non_empty)
                         and any(kw in vals_lower for kw in header_kws))

        if is_tbl_header:
            current_header = [v for v in row_vals]
            continue

        # Baris data dengan header aktif → natural language
        if current_header and len(current_header) == len(row_vals):
            parts = []
            for col_h, val in zip(current_header, row_vals):
                if (val and val.lower() not in ("nan","none","")
                        and col_h and col_h.lower() not in ("nan","none","")):
                    parts.append(f"{col_h}: {val}")
            if parts:
                lines.append(" | ".join(parts))
        else:
            # Baris data tanpa header yang jelas → dump semua nilai
            lines.append(" | ".join(non_empty))

    return "\n".join(lines)

def extract_text(file_path: Path) -> str:
    text   = ""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_text()
                if extracted:
                    # Bersihkan duplikasi teks (umum di PDF desain/layout)
                    lines = extracted.split("\n")
                    seen, clean = set(), []
                    for line in lines:
                        stripped = line.strip()
                        if stripped and stripped not in seen:
                            seen.add(stripped)
                            clean.append(line)
                    page_text = "\n".join(clean).strip()
                    if page_text:
                        # Tandai tiap halaman sebagai unit tersendiri
                        text += f"\n\n=== HALAMAN {page_num+1} ===\n{page_text}\n"
    elif suffix in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif suffix in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            if df.empty:
                continue
            # Gunakan extractor khusus kalender untuk sheet dengan nama "calendar"
            if "calendar" in sheet_name.lower():
                text += extract_calendar_text(df, file_path.stem) + "\n\n"
            else:
                text += excel_sheet_to_text(df, sheet_name, file_path.stem) + "\n\n"
    elif suffix == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                df   = pd.read_csv(file_path, encoding=enc).fillna("")
                text = dataframe_to_text(df, file_path.stem)
                break
            except Exception:
                continue
    return text.strip()


def semantic_chunk(text: str, source_name: str) -> list[dict]:
    """
    Semantic chunking dengan page-aware splitting.
    Prioritas: potong di page boundary (=== HALAMAN N ===) dulu,
    lalu potong berdasarkan paragraf jika halaman terlalu panjang.
    """
    chunks = []
    idx    = 0

    # Pisah berdasarkan page boundary dulu
    page_sections = re.split(r'(=== HALAMAN \d+ ===)', text.strip())

    # Gabungkan marker dengan kontennya
    pages = []
    i = 0
    while i < len(page_sections):
        if re.match(r'=== HALAMAN \d+ ===', page_sections[i].strip()):
            # Header halaman + kontennya
            header  = page_sections[i].strip()
            content_part = page_sections[i+1].strip() if i+1 < len(page_sections) else ""
            if content_part:
                pages.append(f"{header}\n{content_part}")
            i += 2
        else:
            # Teks sebelum halaman pertama (intro)
            intro = page_sections[i].strip()
            if intro:
                pages.append(intro)
            i += 1

    for page_text in pages:
        if len(page_text) <= CHUNK_SIZE:
            # Halaman muat dalam satu chunk
            if len(page_text) > 50:
                chunks.append({
                    "id":     f"{source_name}__chunk_{idx}",
                    "text":   page_text,
                    "source": source_name
                })
                idx += 1
        else:
            # Halaman terlalu panjang — potong berdasarkan paragraf
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', page_text)
                         if p.strip() and len(p.strip()) > 20]
            buffer = ""
            for para in paragraphs:
                candidate = (buffer + "\n\n" + para).strip() if buffer else para
                if len(candidate) <= CHUNK_SIZE:
                    buffer = candidate
                else:
                    if buffer and len(buffer) > 50:
                        chunks.append({
                            "id":     f"{source_name}__chunk_{idx}",
                            "text":   buffer,
                            "source": source_name
                        })
                        idx += 1
                    # Overlap kecil
                    overlap = buffer[-CHUNK_OVERLAP:] if buffer and len(buffer) > CHUNK_OVERLAP else ""
                    buffer  = (overlap + "\n\n" + para).strip() if overlap else para

                    # Potong paksa jika paragraf sangat panjang
                    if len(buffer) > CHUNK_SIZE * 2:
                        start = 0
                        while start < len(buffer):
                            end   = min(start + CHUNK_SIZE, len(buffer))
                            piece = buffer[start:end].strip()
                            if len(piece) > 50:
                                chunks.append({
                                    "id":     f"{source_name}__chunk_{idx}",
                                    "text":   piece,
                                    "source": source_name
                                })
                                idx += 1
                            start += CHUNK_SIZE - CHUNK_OVERLAP
                        buffer = ""

            if buffer and len(buffer) > 50:
                chunks.append({
                    "id":     f"{source_name}__chunk_{idx}",
                    "text":   buffer,
                    "source": source_name
                })
                idx += 1

    return chunks


def chunk_text(text: str, source_name: str) -> list[dict]:
    """Wrapper — gunakan semantic chunking untuk semua dokumen."""
    return semantic_chunk(text, source_name)


def index_documents(collection) -> int:
    supported = {".pdf", ".docx", ".doc", ".txt"}
    files     = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in supported]
    indexed   = 0
    for file_path in files:
        source_name = file_path.stem
        if collection.get(where={"source": source_name})["ids"]:
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
def retrieve(collection, query: str, reranker=None) -> tuple[str, list[str]]:
    """
    Hybrid RAG pipeline:
    1. Dense retrieval: ambil TOP_K_RETRIEVE kandidat dari vector store
    2. Reranking: CrossEncoder memberi skor relevansi yang lebih akurat
    3. Ambil TOP_K_RERANK chunk terbaik sebagai konteks final
    """
    # Step 1: Dense retrieval — ambil banyak kandidat dulu
    results   = collection.query(query_texts=[query], n_results=TOP_K_RETRIEVE)
    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    sources   = sorted({m["source"] for m in metas})

    if not chunks:
        return "", []

    # Step 2: Reranking dengan CrossEncoder
    if reranker is not None:
        try:
            # CrossEncoder menilai relevansi setiap (query, chunk) pair
            pairs  = [[query, chunk] for chunk in chunks]
            scores = reranker.predict(pairs)

            # Urutkan berdasarkan skor tertinggi
            ranked = sorted(zip(scores, chunks, metas), key=lambda x: x[0], reverse=True)

            # Ambil TOP_K_RERANK terbaik
            top_chunks = [chunk for _, chunk, _ in ranked[:TOP_K_RERANK]]
            top_metas  = [meta  for _, _, meta  in ranked[:TOP_K_RERANK]]
            sources    = sorted({m["source"] for m in top_metas})

        except Exception:
            # Fallback ke top chunks tanpa reranking
            top_chunks = chunks[:TOP_K_RERANK]
    else:
        # Tanpa reranker: filter berdasarkan distance, ambil top 3
        distances = results["distances"][0]
        filtered  = [(c, d) for c, d in zip(chunks, distances) if d < 1.5]
        if not filtered:
            filtered = list(zip(chunks, distances))
        top_chunks = [c for c, _ in filtered[:TOP_K_RERANK]]

    context = "\n\n---\n\n".join(top_chunks)
    return context, sources


def generate_answer(client: Groq, query: str, context: str) -> str:
    system_prompt = """Kamu adalah OrmawAI, sebuah sistem chatbot pencari informasi dari dokumen organisasi kemahasiswaan.

IDENTITAS KAMU:
- Kamu adalah SISTEM AI, bukan manusia, bukan anggota BEM, bukan pengurus ormawa manapun.
- Kamu TIDAK memiliki nama, jabatan, atau afiliasi organisasi apapun.
- Jika ada teks di dokumen yang menggunakan kata "saya" atau "kami", itu adalah ucapan ORANG LAIN yang kamu kutip, bukan ucapanmu sendiri.

ATURAN UTAMA — WAJIB DIIKUTI:
1. Jawab HANYA berdasarkan teks yang tersedia di bagian KONTEKS di bawah.
2. DILARANG KERAS menyimpulkan, menginterpretasi, atau mengarang informasi yang tidak tertulis eksplisit di konteks.
3. Jika teks visi/misi/nilai/prosedur tersedia di konteks, kutip langsung kata per kata — jangan parafrase.
4. Jika informasi yang ditanyakan TIDAK ADA di konteks, jawab tepat satu kalimat: "Informasi ini tidak tersedia dalam dokumen yang ada."
5. JANGAN pernah menjawab dengan kata "saya" merujuk pada dirimu sendiri sebagai orang yang punya jabatan atau nama.
6. Ketika mengutip teks yang berisi kata "saya" dari dokumen, ganti dengan nama orang yang berbicara atau gunakan frasa "menurut beliau".

Format jawaban:
- Jawab langsung tanpa basa-basi pembuka
- Gunakan bullet atau nomor jika informasi berupa daftar
- Jangan sebut nama dokumen sumber di dalam jawaban
- Bahasa Indonesia yang natural dan mudah dipahami"""

    user_prompt = f"""Pertanyaan: {query}

KONTEKS (jawab HANYA dari teks di bawah ini, ingat kamu adalah SISTEM AI bukan orang di dalam dokumen):
{context}

Jawab pertanyaan di atas secara objektif sebagai sistem AI. Jika ada kata "saya" di konteks, itu adalah ucapan orang lain — kutip dengan menyebut nama orangnya. Jika jawaban tidak ada di konteks, katakan tidak tersedia."""

    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature = 0.0,
        max_tokens  = 1024
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────
#  UI
# ─────────────────────────────────────────
def main():
    st.set_page_config(
        page_title = "OrmawAI",
        page_icon  = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><text y='20' font-size='20'>🎓</text></svg>",
        layout     = "wide",
        initial_sidebar_state = "expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Fix: force sidebar selalu expanded, sembunyikan tombol collapse bawaan
    st.markdown("""
    <style>
    /* Sembunyikan semua varian tombol collapse Streamlit */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    button[aria-label="Close sidebar"] { display: none !important; }
    button[aria-label="Collapse sidebar"] { display: none !important; }
    /* Pastikan sidebar selalu visible */
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: flex !important;
        transform: none !important;
        min-width: 280px !important;
    }
    </style>
    <script>
    // Buka sidebar jika tertutup
    function forceOpenSidebar() {
        const btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false' && btn) {
            btn.click();
        }
    }
    setTimeout(forceOpenSidebar, 200);
    setTimeout(forceOpenSidebar, 800);
    </script>
    """, unsafe_allow_html=True)

    collection  = init_chromadb()
    groq_client = init_groq()
    reranker    = init_reranker()

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        # Logo + judul
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <div style="width:36px;height:36px;background:rgba(37,99,235,0.15);border:1px solid rgba(37,99,235,0.3);
                        border-radius:10px;display:flex;align-items:center;justify-content:center">
                {icon("graduation", 18, "#63b3ed")}
            </div>
            <div>
                <div style="font-size:1rem;font-weight:600;color:#f0f6fc;line-height:1.2">OrmawAI</div>
                <div style="font-size:0.7rem;color:rgba(255,255,255,0.35)">v1.0 · RAG + Groq</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Upload — styled native file uploader (drag & drop bawaan Streamlit)
        st.markdown(f'<div class="sec-label">{icon("upload",13,"rgba(255,255,255,0.35)")} Upload Dokumen</div>', unsafe_allow_html=True)

        # Styling agresif pada native Streamlit file_uploader
        # agar tampil seperti custom drag & drop zone
        st.markdown("""
        <style>
        /* Container utama */
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed rgba(99,179,237,0.35) !important;
            border-radius: 12px !important;
            background: rgba(37,99,235,0.04) !important;
            padding: 0 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: rgba(99,179,237,0.65) !important;
            background: rgba(37,99,235,0.08) !important;
        }
        /* Dropzone area */
        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            padding: 20px 16px !important;
            cursor: pointer !important;
        }
        /* Sembunyikan ikon default dan teks bawaan */
        [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
            display: none !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            display: none !important;
        }
        /* Ganti ikon dengan SVG upload custom via pseudo-element */
        [data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "";
            display: block;
            width: 28px;
            height: 28px;
            margin: 0 auto 10px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='rgba(99,179,237,0.65)' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: center;
        }
        /* Ganti teks instructions */
        [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "Drag & drop dokumen di sini";
            display: block;
            font-size: 0.84rem;
            font-weight: 500;
            color: rgba(255,255,255,0.65);
            text-align: center;
            margin-bottom: 4px;
        }
        /* Tombol Browse */
        [data-testid="stFileUploaderDropzone"] button {
            background: rgba(37,99,235,0.15) !important;
            border: 1px solid rgba(37,99,235,0.35) !important;
            color: #93c5fd !important;
            border-radius: 7px !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            padding: 5px 14px !important;
            margin-top: 6px !important;
            transition: all 0.15s !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: rgba(37,99,235,0.28) !important;
            border-color: rgba(99,179,237,0.5) !important;
        }
        /* Format label di bawah tombol */
        [data-testid="stFileUploader"] section > div > div > small {
            font-size: 0.7rem !important;
            color: rgba(255,255,255,0.28) !important;
        }
        /* File yang sudah diupload */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 8px !important;
            padding: 6px 10px !important;
            margin-top: 6px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            label="Drag & drop atau klik untuk pilih file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Format: PDF, DOCX, TXT · Maks. 200MB/file"
        )

        if uploaded_files:
            DOCS_DIR.mkdir(exist_ok=True)
            for uf in uploaded_files:
                (DOCS_DIR / uf.name).write_bytes(uf.read())
            with st.spinner("Mengindeks dokumen..."):
                n = index_documents(collection)
            if n > 0:
                st.success(f"✅ {n} chunk diindeks dari {len(uploaded_files)} file!")
            else:
                st.info("Semua dokumen sudah terindeks.")


        st.divider()

        # Daftar dokumen
        st.markdown(f'<div class="sec-label">{icon("layers",13,"rgba(255,255,255,0.35)")} Dokumen Terindeks</div>', unsafe_allow_html=True)
        all_docs = collection.get()
        if all_docs["ids"]:
            sources  = sorted({m["source"] for m in all_docs["metadatas"]})
            n_chunks = len(all_docs["ids"])
            for src in sources:
                st.markdown(
                    f'<div class="stat-chip"><div class="dot-green"></div>'
                    f'{icon("file",14,"rgba(255,255,255,0.5)")} {src}</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<p style="font-size:0.72rem;color:rgba(255,255,255,0.28);margin-top:8px">'
                f'{len(sources)} dokumen · {n_chunks} chunk</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="display:flex;gap:8px;align-items:flex-start;padding:10px 0">'
                f'{icon("info",15,"rgba(255,255,255,0.3)")}'
                f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.3)">Belum ada dokumen.<br>Upload di atas.</span></div>',
                unsafe_allow_html=True
            )

        st.divider()

        # Reset
        if st.button(f"Reset semua dokumen", use_container_width=True, key="reset_btn"):
            hard_reset_vectorstore()
            st.success("✅ Reset berhasil! Silakan upload dokumen baru.")
            st.rerun()
        st.markdown(f'<p style="font-size:0.7rem;color:rgba(255,255,255,0.2);text-align:center;margin-top:4px">Hapus semua dokumen dari vector store</p>', unsafe_allow_html=True)

    # ── MAIN ─────────────────────────────────────────────────────

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-top">
            <div class="hero-icon">{icon("graduation", 20, "rgba(255,255,255,0.7)")}</div>
            <h1>OrmawAI</h1>
        </div>
        <p class="hero-sub">
            Tanyakan apa saja tentang dokumen organisasi kamu.
            Dijawab langsung dari dokumen yang kamu upload.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_chat, col_hist = st.columns([3, 1], gap="large")

    with col_chat:
        # Input
        st.markdown(
            f'<div class="sec-label">{icon("book",13,"rgba(255,255,255,0.3)")} Pertanyaan Kamu</div>',
            unsafe_allow_html=True
        )
        query = st.text_area(
            label="q", value=st.session_state.get("prefill",""),
            height=110,
            placeholder="Contoh: Bagaimana prosedur pengajuan proposal kegiatan BEM?",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        cari = st.button(f"Cari Jawaban", type="primary", use_container_width=True, key="cari_btn")

        if cari:
            st.session_state.pop("prefill", None)
            if not query.strip():
                st.warning("Masukkan pertanyaan terlebih dahulu.")
            elif not collection.get()["ids"]:
                st.warning("Belum ada dokumen. Upload dokumen BEM di sidebar kiri.")
            else:
                prog = st.progress(0, text="Mencari kandidat dokumen relevan...")
                context, sources = retrieve(collection, query, reranker)
                prog.progress(55, text="Menyusun jawaban dengan AI...")
                answer = generate_answer(groq_client, query, context)
                prog.progress(100, text="Selesai!")
                prog.empty()

                # Jawaban
                st.markdown(
                    f'<div class="sec-label" style="margin-top:20px">'
                    f'{icon("check",13,"rgba(255,255,255,0.3)")} Jawaban</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # Sumber
                st.markdown(
                    f'<div class="sec-label">'
                    f'{icon("file",13,"rgba(255,255,255,0.3)")} Sumber Dokumen</div>',
                    unsafe_allow_html=True
                )
                badges = "".join([
                    f'<span class="src-badge">{icon("file",12,"#93c5fd")} &nbsp;{s}</span>'
                    for s in sources
                ])
                st.markdown(badges, unsafe_allow_html=True)

                # Riwayat
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].insert(0, {"q": query, "a": answer, "sources": sources})

    with col_hist:
        st.markdown(
            f'<div class="sec-label">{icon("clock",13,"rgba(255,255,255,0.3)")} Riwayat</div>',
            unsafe_allow_html=True
        )
        history = st.session_state.get("history", [])
        if not history:
            st.markdown(
                '<p style="font-size:0.81rem;color:rgba(255,255,255,0.28);padding-top:4px">'
                'Belum ada riwayat pertanyaan.</p>',
                unsafe_allow_html=True
            )
        else:
            for item in history[:6]:
                short_q = item["q"][:55] + ("..." if len(item["q"]) > 55 else "")
                short_a = item["a"][:130] + ("..." if len(item["a"]) > 130 else "")
                src_str = " · ".join(item["sources"])
                st.markdown(f"""
                <div class="hist-card">
                    <div class="hist-q">{icon("search",11,"rgba(255,255,255,0.3)")} &nbsp;{short_q}</div>
                    <div class="hist-a">{short_a}</div>
                    <div class="hist-src">{icon("file",10,"rgba(255,255,255,0.2)")} &nbsp;{src_str}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()