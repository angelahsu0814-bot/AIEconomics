import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from fredapi import Fred
from google import genai
from datetime import datetime

# ==========================================
# 1. 頁面基本配置
# ==========================================

st.set_page_config(
    page_title="總體經濟與金融即時儀表板",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 全球總體經濟與金融指標即時儀表板")
st.caption("整合 FRED、Yahoo Finance 與 Google Gemini AI")

# ==========================================
# 2. 側邊欄：API Key 設定
# ==========================================

st.sidebar.header("⚙️ 設定與金鑰")

# 安全讀取 Streamlit Secrets
def read_secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

# FRED API Key
fred_secret = read_secret("FRED_API_KEY")

if fred_secret:
    fred_key = fred_secret
    st.sidebar.success("FRED API Key 已從 Secrets 讀取")
else:
    fred_key = st.sidebar.text_input(
        "FRED API Key",
        type="password"
    )

# Gemini API Key
gemini_secret = read_secret("GEMINI_API_KEY")

if gemini_secret:
    gemini_key = gemini_secret
    st.sidebar.success("Gemini API Key 已從 Secrets 讀取")
else:
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password"
    )

# 手動刷新數據
if st.sidebar.button("🔄 手動刷新數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 3. 抓取 FRED 總體經濟數據
# 快取時間：1 小時
# ==========================================

@st.cache_data(ttl=3600)
def get_fred_data(api_key):

    fred = Fred(api_key=api_key)

    # 抓取 CPI 原始指數
    cpi = fred.get_series("CPIAUCSL").dropna()

    # 計算 CPI 年增率（Year-over-Year）
    cpi_yoy = cpi.pct_change(12) * 100

    # 其他總體經濟指標
    data = {
        "CPI 物價指數": cpi,
        "CPI 年增率": cpi_yoy,
        "10年期美債殖利率": fred.get_series("DGS10"),
        "美國失業率": fred.get_series("UNRATE"),
        "聯邦基金利率": fred.get_series("FEDFUNDS")
    }

    df = pd.DataFrame(data).sort_index().ffill()

    return df.tail(365)


# ==========================================
# 4. 抓取 Yahoo Finance 金融市場數據
# 快取時間：10 分鐘
# ==========================================

@st.cache_data(ttl=600)
def get_market_data():

    tickers = {
        "S&P 500": "^GSPC",
        "台灣加權指數": "^TWII",
        "布蘭特原油": "BZ=F",
        "美元/台幣匯率": "TWD=X"
    }

    raw = yf.download(
        list(tickers.values()),
        period="1y",
        auto_adjust=False,
        progress=False
    )

    if raw.empty:
        raise ValueError("Yahoo Finance 沒有回傳市場數據")

    # 處理 yfinance 不同版本的欄位格式
    if isinstance(raw.columns, pd.MultiIndex):

        if "Close" in raw.columns.get_level_values(0):
            market_df = raw["Close"]

        else:
            market_df = raw.xs(
                "Close",
                axis=1,
                level=1
            )

    else:
        market_df = raw[["Close"]].copy()

    market_df.rename(
        columns={
            ticker: name
            for name, ticker in tickers.items()
        },
        inplace=True
    )

    return market_df.dropna(how="all")


# ==========================================
# 5. 取得最新數值與日期
# ==========================================

def get_latest_value(df, column):

    series = df[column].dropna()

    if series.empty:
        raise ValueError(f"{column} 沒有可用數據")

    return {
        "value": float(series.iloc[-1]),
        "date": series.index[-1].strftime("%Y-%m-%d")
    }


# ==========================================
# 6. Gemini AI 總體經濟分析函數
# ==========================================

def generate_economic_summary(data, api_key):

    # 建立 Google Gemini Client
    client = genai.Client(api_key=api_key)

    # 整理最新經濟數據
    economic_context = f"""
    以下是最新取得的美國總體經濟數據：

    1. CPI 物價指數：
       {data['CPI']['value']:.2f}
       資料日期：{data['CPI']['date']}

    2. CPI 年增率：
       {data['CPI_YOY']['value']:.2f}%
       資料日期：{data['CPI_YOY']['date']}

    3. 美國失業率：
       {data['UNEMPLOYMENT']['value']:.2f}%
       資料日期：{data['UNEMPLOYMENT']['date']}

    4. 美國10年期公債殖利率：
       {data['TREASURY']['value']:.2f}%
       資料日期：{data['TREASURY']['date']}

    5. 美國聯邦基金利率：
       {data['FED_RATE']['value']:.2f}%
       資料日期：{data['FED_RATE']['date']}
    """

    # AI 分析指令
    prompt = f"""
    你是一位專業的總體經濟分析師，
    請根據以下經濟數據撰寫一篇約200字的繁體中文總體經濟短評。

    【分析要求】

    一、物價與通貨膨脹：
    分析 CPI 指數及 CPI 年增率，
    說明目前物價與通膨狀況。

    二、勞動市場：
    根據美國失業率，分析就業市場狀況。

    三、貨幣政策與利率：
    分析10年期美債殖利率及聯邦基金利率，
    說明利率對消費、投資及企業融資的可能影響。

    四、整體經濟展望：
    綜合以上指標，評估美國目前的總體經濟情勢，
    並提出簡短的未來觀察方向。

    【撰寫規則】

    1. 使用繁體中文。
    2. 全文約200字。
    3. 使用正式但容易理解的經濟學用語。
    4. 不可以自行捏造數據。
    5. 不可將單一時點數值直接解讀成上升或下降趨勢。
    6. 不可將相關性直接解讀為因果關係。
    7. 注意不同經濟指標的資料日期可能不同。
    8. 請直接輸出完整短評，不需要列出分析步驟。

    【最新經濟數據】

    {economic_context}
    """

    # 呼叫 Gemini API
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "temperature": 0.4,
            "max_output_tokens": 600
        }
    )

    if not response.text:
        raise ValueError("Gemini 沒有產生文字內容")

    return response.text


# ==========================================
# 7. 主頁面：抓取及呈現數據
# ==========================================

if fred_key:

    try:

        macro_df = get_fred_data(fred_key)
        market_df = get_market_data()

        # 取得最新數值
        latest_cpi = get_latest_value(
            macro_df, "CPI 物價指數"
        )

        latest_cpi_yoy = get_latest_value(
            macro_df, "CPI 年增率"
        )

        latest_rate = get_latest_value(
            macro_df, "10年期美債殖利率"
        )

        latest_unemp = get_latest_value(
            macro_df, "美國失業率"
        )

        latest_fed = get_latest_value(
            macro_df, "聯邦基金利率"
        )

        # ==================================
        # 第一區塊：核心 KPI 卡片
        # ==================================

        st.subheader("📊 關鍵總體經濟指標 (FRED)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "CPI 物價指數",
                f"{latest_cpi['value']:.1f}"
            )
            st.caption(
                f"資料日期：{latest_cpi['date']}"
            )

        with col2:

            rate_series = macro_df[
                "10年期美債殖利率"
            ].dropna()

            rate_change = (
                rate_series.iloc[-1] -
                rate_series.iloc[-2]
            )

            st.metric(
                "10年期美債殖利率",
                f"{latest_rate['value']:.2f}%",
                f"{rate_change:+.2f} 個百分點"
            )

            st.caption(
                f"資料日期：{latest_rate['date']}"
            )

        with col3:
            st.metric(
                "美國失業率",
                f"{latest_unemp['value']:.1f}%"
            )

            st.caption(
                f"資料日期：{latest_unemp['date']}"
            )

        with col4:
            st.metric(
                "聯邦基金利率",
                f"{latest_fed['value']:.2f}%"
            )

            st.caption(
                f"資料日期：{latest_fed['date']}"
            )

        # 額外顯示 CPI 年增率
        st.info(
            f"目前 CPI 年增率："
            f"{latest_cpi_yoy['value']:.2f}%"
            f"（資料日期：{latest_cpi_yoy['date']}）"
        )

        # ==================================
        # 第二區塊：互動式圖表
        # ==================================

        st.write("---")

        st.subheader("📈 市場走勢與總體指標折線圖")

        tab1, tab2 = st.tabs([
            "金融市場走勢 (yfinance)",
            "總體經濟趨勢 (FRED)"
        ])

        with tab1:

            if not market_df.empty:

                market_choice = st.selectbox(
                    "選擇金融市場標的：",
                    market_df.columns
                )

                fig1 = px.line(
                    market_df,
                    y=market_choice,
                    title=f"{market_choice} 近一年走勢"
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

            else:
                st.warning("目前沒有可用的金融市場數據")

        with tab2:

            macro_choice = st.selectbox(
                "選擇總體經濟指標：",
                macro_df.columns
            )

            fig2 = px.line(
                macro_df,
                y=macro_choice,
                title=f"{macro_choice} 近一年走勢"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # ==================================
        # 第三區塊：AI 總體經濟摘要
        # ==================================

        st.write("---")

        st.subheader("🤖 AI 總體經濟分析")

        st.write(
            "利用 Google Gemini AI，"
            "根據最新 CPI、失業率及利率數據，"
            "自動生成約200字的總體經濟短評。"
        )

        # 準備 AI 分析所需數據
        ai_data = {
            "CPI": latest_cpi,
            "CPI_YOY": latest_cpi_yoy,
            "UNEMPLOYMENT": latest_unemp,
            "TREASURY": latest_rate,
            "FED_RATE": latest_fed
        }

        # AI 摘要按鈕
        if st.button(
            "✨ 產生 AI 總體經濟摘要",
            use_container_width=True,
            type="primary"
        ):

            if not gemini_key:

                st.error(
                    "請先在左側設定 Gemini API Key！"
                )

            else:

                with st.spinner(
                    "Gemini AI 正在分析最新經濟數據，請稍候..."
                ):

                    try:

                        summary = generate_economic_summary(
                            ai_data,
                            gemini_key
                        )

                        st.success(
                            "AI 總體經濟摘要產生成功！"
                        )

                        st.markdown(
                            "### 📋 美國總體經濟短評"
                        )

                        st.info(summary)

                        st.caption(
                            "分析依據：FRED 最新可取得數據。"
                            "AI 生成內容僅供學術研究參考。"
                        )

                    except Exception as e:

                        st.error(
                            f"AI 摘要產生失敗：{e}"
                        )

        st.write("---")

        st.caption(
            "資料來源：FRED、Yahoo Finance、Google Gemini AI"
        )

        st.caption(
            f"儀表板執行時間："
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except Exception as e:

        st.error(
            f"數據讀取失敗，請確認 API Key 或資料來源。"
            f"錯誤訊息：{e}"
        )

else:

    st.info(
        "👈 請在左側輸入 FRED API Key 以開啟儀表板。"
    )
