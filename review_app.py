"""Small Streamlit review UI over the SQLite store.

Run:  streamlit run review_app.py -- --db yolo26_seg_test/race.db

Lets a human scan crossing events (defaulting to the ones the pipeline flagged
for review), see the segmented candidate crops inline, and confirm or override
the bib into the verified_bib column. Reads run_dir from the runs table so it can
resolve the candidate PNGs saved by the pipeline.
"""

import argparse
import sys
from pathlib import Path

import streamlit as st

import db
from common import digits_only

DEFAULT_WATCH = "3544,3460,4315,4761,16565,4163"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="yolo26_seg_test/race.db")
    # Streamlit passes script args after "--"; ignore anything it injects.
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource
def get_conn(db_path):
    return db.connect(db_path)


def candidate_pngs(run_dir, candidate_dir):
    folder = Path(run_dir) / "ocr_candidates" / (candidate_dir or "")
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.png"))


def main():
    args = get_args()
    st.set_page_config(page_title="Bib Review", layout="wide")
    st.title("🏁 Race Bib Review")

    if not Path(args.db).exists():
        st.error(f"DB not found: {args.db}. Run the pipeline with --db or ingest_csv_to_db.py first.")
        return
    conn = get_conn(args.db)

    runs = conn.execute("SELECT run_id, run_dir, crossing_events FROM runs ORDER BY run_id DESC").fetchall()
    if not runs:
        st.warning("No runs in the database yet.")
        return

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    run_labels = {f"#{r['run_id']}  {Path(r['run_dir']).name}  ({r['crossing_events']} events)": r for r in runs}
    run_choice = st.sidebar.selectbox("Run", list(run_labels))
    run = run_labels[run_choice]
    run_id, run_dir = run["run_id"], run["run_dir"]

    status_filter = st.sidebar.selectbox("Annotation status", ["review", "accepted", "all"], index=0)
    only_watch = st.sidebar.checkbox("Only watchlist bibs", value=False)
    watch = {digits_only(b) for b in st.sidebar.text_input("Watchlist bibs", DEFAULT_WATCH).split(",") if digits_only(b)}
    hide_verified = st.sidebar.checkbox("Hide already-verified", value=False)

    where = ["run_id = ?"]
    params = [run_id]
    if status_filter != "all":
        where.append("annotation_status = ?")
        params.append(status_filter)
    if hide_verified:
        where.append("verified_bib IS NULL")
    rows = conn.execute(
        f"SELECT * FROM crossings WHERE {' AND '.join(where)} ORDER BY CAST(frame AS INTEGER)",
        params,
    ).fetchall()
    if only_watch:
        rows = [r for r in rows if (r["matched_bib"] in watch or r["ocr_digits"] in watch)]

    # --- Summary ---
    all_rows = conn.execute("SELECT annotation_status, verified_bib FROM crossings WHERE run_id = ?", (run_id,)).fetchall()
    accepted = sum(1 for r in all_rows if r["annotation_status"] == "accepted")
    review = sum(1 for r in all_rows if r["annotation_status"] == "review")
    verified = sum(1 for r in all_rows if r["verified_bib"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(all_rows))
    c2.metric("Accepted", accepted)
    c3.metric("Needs review", review)
    c4.metric("Verified", verified)

    st.caption(f"Showing {len(rows)} event(s) from {Path(run_dir).name}")

    # --- Table ---
    table = [{
        "id": r["id"], "frame": r["frame"], "time_sec": r["time_sec"], "track_id": r["track_id"],
        "ocr_digits": r["ocr_digits"], "matched_bib": r["matched_bib"], "match_type": r["match_type"],
        "match_probability": r["match_probability"], "status": r["annotation_status"],
        "review_reason": r["review_reason"], "verified_bib": r["verified_bib"],
    } for r in rows]
    st.dataframe(table, use_container_width=True, hide_index=True)

    if not rows:
        return

    # --- Per-event review panel ---
    st.subheader("Review an event")
    by_id = {r["id"]: r for r in rows}
    label = {f"id {r['id']} · frame {r['frame']} · track {r['track_id']} · ocr={r['ocr_digits'] or '—'} · bib={r['matched_bib'] or '—'}": r["id"] for r in rows}
    chosen = st.selectbox("Event", list(label))
    row = by_id[label[chosen]]

    left, right = st.columns([2, 1])
    with left:
        pngs = candidate_pngs(run_dir, row["candidate_dir"])
        best = row["best_crop"]
        best_pngs = [p for p in pngs if p.name == best]
        other = [p for p in pngs if p.name != best]
        if best_pngs:
            st.markdown("**Best candidate crop**")
            st.image(str(best_pngs[0]), width=220)
        if other:
            st.markdown(f"**Backlog ({len(other)})**")
            st.image([str(p) for p in other], width=150)
        if not pngs:
            st.info(f"No candidate crops found under {run_dir}/ocr_candidates/{row['candidate_dir']}")

    with right:
        st.markdown("**Confirm bib**")
        st.write(f"OCR digits: `{row['ocr_digits'] or '—'}`")
        st.write(f"Matched bib: `{row['matched_bib'] or '—'}`  ({row['match_type'] or 'no match'})")
        st.write(f"Reason: {row['review_reason'] or '—'}")
        default = row["verified_bib"] or row["matched_bib"] or row["ocr_digits"] or ""
        value = st.text_input("verified_bib", default, key=f"vb_{row['id']}")
        note = st.text_input("note", row["verify_note"] or "", key=f"nt_{row['id']}")
        col_a, col_b = st.columns(2)
        if col_a.button("✅ Confirm", key=f"ok_{row['id']}"):
            db.set_verified_bib(conn, row["id"], digits_only(value) or value, by="ui", note=note)
            st.success(f"Saved verified_bib = {value}")
            st.rerun()
        if col_b.button("Same as OCR", key=f"same_{row['id']}"):
            db.set_verified_bib(conn, row["id"], row["ocr_digits"], by="ui", note="quick-accept OCR")
            st.rerun()


if __name__ == "__main__":
    main()
