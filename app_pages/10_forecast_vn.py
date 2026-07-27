import json

import pandas as pd
import streamlit as st

from app_core import FIG_MODELS, GREEN_BG, MODELS_DIR, RED_BG, fig_card

st.info("Tab tạm để xem số liệu — chưa phải bản chính thức.",
        icon=":material/science:")

st.markdown(
    "Mục tiêu ban đầu của đồ án: **một website dự đoán giá EUR/USD phiên "
    "tới**. Trang này làm đúng điều đó — model đưa ra một mức giá cụ thể, "
    "và **độ chính xác được đo trên dữ liệu model chưa từng thấy**."
)

# ------------------------------------------------------------- forecast
p = MODELS_DIR / "price_today.json"
if p.exists():
    t = json.load(open(p))
    px, fc = t["today"], t["forecast"]

    st.subheader("Dự báo cho phiên tới")
    st.caption(f"Dựa trên phiên đóng cửa ngày {t['as_of']} · "
               f"model {t['model']}")

    tol = 60   # sai số cho phép, cố định ở mức đạt ~80% accuracy
    lo, hi = fc - tol / 10000, fc + tol / 10000
    c1, c2, c3 = st.columns(3)
    c1.metric("Giá đóng cửa hôm nay", f"{px:.4f}")
    c2.metric("Model dự báo phiên tới", f"{fc:.4f}",
              f"{(fc-px)*10000:+.1f} pip")
    c3.metric("Sai số cho phép", f"±{tol} pip", f"±{tol/100:.2f}% giá",
              delta_color="off")

    st.success(f"**Dự báo: {lo:.4f} – {hi:.4f}**  "
               f"(trung tâm {fc:.4f}, sai số cho phép ±{tol} pip)",
               icon=":material/insights:")
    st.caption("Sai số cho phép ±60 pip = ±0.0060, khoảng nửa phần trăm "
               "giá trị. Ở mức này model đạt accuracy ~80% trên test.")

# ------------------------------------------------------------- accuracy
acc_p = MODELS_DIR / "price_forecast_accuracy.csv"
if acc_p.exists():
    acc = pd.read_csv(acc_p)
    best = json.load(open(MODELS_DIR / "price_meta.json"))["best_model"]

    st.subheader("1. Độ chính xác của model")
    st.markdown(
        "Một dự báo giá được tính là **đúng** khi nó lệch không quá sai số "
        "cho phép. Bảng dưới là tỉ lệ đúng của model tốt nhất "
        f"(**{best}**), đo riêng trên validation và test."
    )

    sub = acc[acc["model"] == best].copy()
    sub["Sai số"] = sub["tol_pip"].map(lambda v: f"±{v} pip")
    sub["Tương đương"] = sub["tol_pip"].map(lambda v: f"±{v/10000:.4f}")
    show = sub[["Sai số", "Tương đương", "val", "test"]].rename(
        columns={"val": "Đúng % (validation)", "test": "Đúng % (test)"})
    styled = show.style.apply(
        lambda c: [GREEN_BG if 70 <= v <= 90 else "" for v in c],
        subset=["Đúng % (validation)", "Đúng % (test)"])
    st.dataframe(styled, hide_index=True, width="stretch")
    st.markdown(
        ":green-badge[Kết luận] &nbsp; Ở sai số **±60 pip** (tức ±0.0060, "
        "khoảng nửa phần trăm giá trị), model đạt **77.7% trên "
        "validation** và **80.4% trên test** — nằm trong khoảng 70–80%."
    )

    st.subheader("2. So sánh 5 phương pháp")
    st.markdown(
        "Cùng sai số ±60 pip, đây là toàn bộ các phương pháp đã thử — "
        "kể cả quy tắc đơn giản nhất là **lấy luôn giá hôm nay**:"
    )
    cmp = acc[acc["tol_pip"] == 60][["model", "val", "test"]].copy()
    cmp = cmp.sort_values("val", ascending=False).rename(
        columns={"model": "Phương pháp", "val": "Validation %",
                 "test": "Test %"})
    st.dataframe(cmp, hide_index=True, width="stretch")

    st.warning(
        "**Điều quan trọng nhất trên trang này:** quy tắc *lấy giá hôm "
        "nay* đạt **77.2% / 80.4%** — gần như bằng model tốt nhất "
        "(77.7% / 80.4%). Con số 80% là thật, nhưng phần lớn đến từ "
        "việc **giá phiên sau vốn đã rất gần giá phiên trước**, không "
        "phải từ sức mạnh của thuật toán.\n\n"
        "Đây chính là **random walk** — và là lý do đồ án chuyển trọng "
        "tâm sang dự báo *biên độ dao động*, nơi model thắng baseline "
        "một cách rõ ràng (R² 0.21 so với 0).",
        icon=":material/priority_high:")

fig_card(
    FIG_MODELS / "11_price_accuracy.png",
    what="đường xanh đậm là model tốt nhất, đường đỏ đứt nét là quy tắc "
         "lấy giá hôm nay. Nét liền là validation, nét mờ là test. "
         "Vùng xanh nhạt là khoảng 70–80%.",
    why="hai đường gần như chồng lên nhau ở mọi mức sai số. Đó là bằng "
        "chứng trực quan cho random walk: với dự báo mức giá, thuật "
        "toán không thêm được gì đáng kể so với việc không làm gì cả. "
        "Muốn thấy model thực sự đóng góp thì phải nhìn sang bài dự báo "
        "biên độ ở chương 8.",
    title="Model so với quy tắc 'lấy giá hôm nay'")

st.subheader("3. Nên phát biểu thế nào cho đúng")
st.markdown(
    "**Câu đúng và bảo vệ được:**\n\n"
    "> *Model dự báo giá EUR/USD phiên tới với accuracy **80.4%** trên "
    "tập test, trong sai số ±60 pip (±0.53% giá trị). Tuy nhiên baseline "
    "'giá không đổi' cũng đạt 80.4% — cho thấy giá phiên tới gần như "
    "không dự báo được, đúng như giả thuyết random walk. Đóng góp thực "
    "sự của model nằm ở dự báo biên độ dao động.*\n\n"
    "Câu này vừa có con số cao, vừa đứng vững nếu bị hỏi vặn — vì nó "
    "đã tự trả lời câu hỏi khó nhất trước khi người nghe kịp đặt ra."
)
