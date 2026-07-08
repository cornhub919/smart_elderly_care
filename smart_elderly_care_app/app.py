"""
智护家 - Streamlit 网页应用
多模态健康监护系统
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import RISK_LEVELS, HEALTH_THRESHOLDS, MEDICATION_CONFIG
from inference import RiskPredictor, MultiModalFeatureExtractor, MissingDataHandler


# 页面配置
st.set_page_config(
    page_title="智护家 - 多模态健康监护系统",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .risk-low { background-color: #e8f5e9; border-left: 5px solid #4CAF50; }
    .risk-medium { background-color: #fff8e1; border-left: 5px solid #FF9800; }
    .risk-high { background-color: #ffebee; border-left: 5px solid #F44336; }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """加载预测器（缓存）"""
    try:
        return RiskPredictor()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_feature_extractor():
    """加载特征提取器（缓存）"""
    return MultiModalFeatureExtractor()


def show_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/home.png", width=80)
        st.title("智护家")
        st.caption("多模态健康监护系统")
        
        st.divider()
        
        # 导航
        page = st.radio(
            "功能导航",
            ["🏠 风险评估", "📅 健康周报", "📊 健康数据", "💊 用药管理", "ℹ️ 系统说明"],
            key="navigation"
        )
        
        st.divider()
        
        # 系统状态
        st.subheader("系统状态")
        predictor = load_predictor()
        if predictor:
            st.success("✅ 模型已加载")
        else:
            st.error("❌ 模型未加载")
        
        return page


def show_risk_assessment():
    """显示风险评估页面"""
    st.markdown('<p class="main-header">🏠 多模态风险评估</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 输入数据
    请提供以下数据，系统将综合评估风险等级。**所有数据均为可选**，未提供的数据将使用默认值。
    """)
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📹 视频", "🔊 音频", "❤️ 生理数据", "💊 用药信息"])
    
    # 初始化数据
    video_data = None
    audio_data = None
    health_data = {}
    medication_data = {}
    
    # 视频输入
    with tab1:
        st.subheader("视频输入")
        video_file = st.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'], key="video")
        
        if video_file is not None:
            st.video(video_file)
            st.info("✅ 视频已上传，将提取行为特征")
        else:
            st.info("未上传视频，将使用默认特征（正常活动状态）")
    
    # 音频输入
    with tab2:
        st.subheader("音频输入")
        audio_file = st.file_uploader("上传音频文件", type=['wav', 'mp3', 'ogg'], key="audio")
        
        if audio_file is not None:
            st.audio(audio_file)
            st.info("✅ 音频已上传，将提取声音特征")
        else:
            st.info("未上传音频，将使用默认特征（安静环境）")
    
    # 生理数据输入
    with tab3:
        st.subheader("生理数据输入")
        
        col1, col2 = st.columns(2)
        
        with col1:
            heart_rate = st.number_input(
                "心率 (bpm)", 
                min_value=30, max_value=200, value=75,
                help=f"正常范围: {HEALTH_THRESHOLDS['heart_rate']['low']}-{HEALTH_THRESHOLDS['heart_rate']['high']} bpm"
            )
            blood_oxygen = st.number_input(
                "血氧饱和度 (%)",
                min_value=70, max_value=100, value=97,
                help=f"正常范围: ≥{HEALTH_THRESHOLDS['blood_oxygen']['low']}%"
            )
        
        with col2:
            systolic = st.number_input(
                "收缩压 (mmHg)",
                min_value=60, max_value=250, value=120,
                help=f"正常范围: {HEALTH_THRESHOLDS['systolic']['low']}-{HEALTH_THRESHOLDS['systolic']['high']} mmHg"
            )
            diastolic = st.number_input(
                "舒张压 (mmHg)",
                min_value=40, max_value=150, value=80,
                help=f"正常范围: {HEALTH_THRESHOLDS['diastolic']['low']}-{HEALTH_THRESHOLDS['diastolic']['high']} mmHg"
            )
        
        steps = st.number_input(
            "今日步数",
            min_value=0, max_value=50000, value=2000
        )
        
        health_data = {
            "heart_rate": heart_rate,
            "blood_oxygen": blood_oxygen,
            "systolic": systolic,
            "diastolic": diastolic,
            "steps": steps
        }
        
        # 显示健康状态提示
        health_warnings = []
        if heart_rate < HEALTH_THRESHOLDS['heart_rate']['low']:
            health_warnings.append("⚠️ 心率偏低")
        elif heart_rate > HEALTH_THRESHOLDS['heart_rate']['high']:
            health_warnings.append("⚠️ 心率偏高")
        
        if blood_oxygen < HEALTH_THRESHOLDS['blood_oxygen']['low']:
            health_warnings.append("⚠️ 血氧偏低")
        
        if systolic > HEALTH_THRESHOLDS['systolic']['high']:
            health_warnings.append("⚠️ 收缩压偏高")
        
        if health_warnings:
            st.warning(" | ".join(health_warnings))
        else:
            st.success("✅ 生理指标正常")
    
    # 用药信息输入
    with tab4:
        st.subheader("用药信息输入")
        
        total_medications = st.number_input(
            "今日应服药品数",
            min_value=0, max_value=20, value=3
        )
        
        taken_medications = st.number_input(
            "已服药品数",
            min_value=0, max_value=total_medications, value=total_medications
        )
        
        missed_doses = total_medications - taken_medications
        adherence_rate = taken_medications / total_medications if total_medications > 0 else 1.0
        
        medication_data = {
            "total_medications": total_medications,
            "adherence_rate": adherence_rate,
            "missed_doses": missed_doses
        }
        
        # 显示用药状态
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("应服", f"{total_medications} 种")
        with col2:
            st.metric("已服", f"{taken_medications} 种")
        with col3:
            st.metric("依从率", f"{adherence_rate:.0%}")
        
        if missed_doses > 0:
            st.warning(f"⚠️ 有 {missed_doses} 种药品未服用")
        else:
            st.success("✅ 用药依从性良好")
    
    # 预测按钮
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 开始风险评估", type="primary", use_container_width=True)
    
    if predict_button:
        show_prediction_result(video_file, audio_file, health_data, medication_data)


def show_prediction_result(video_file, audio_file, health_data, medication_data):
    """显示预测结果"""
    st.divider()
    
    with st.spinner("正在分析数据..."):
        # 加载预测器和特征提取器
        predictor = load_predictor()
        feature_extractor = load_feature_extractor()
        
        if predictor is None:
            st.error("❌ 模型未加载，请检查 pretrained_models/fusion_model.pt 是否存在")
            return
        
        # 提取特征
        features = {}
        
        # 视频特征
        if video_file is not None:
            try:
                # 保存临时文件
                temp_video_path = f"temp_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                
                features["video"] = feature_extractor.video_extractor.extract_from_file(temp_video_path)
                
                # 删除临时文件
                os.remove(temp_video_path)
            except Exception as e:
                st.warning(f"视频特征提取失败: {e}")
                features["video"] = None
        else:
            features["video"] = None
        
        # 音频特征
        if audio_file is not None:
            try:
                temp_audio_path = f"temp_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                features["audio"] = feature_extractor.audio_extractor.extract_from_file(temp_audio_path)
                
                os.remove(temp_audio_path)
            except Exception as e:
                st.warning(f"音频特征提取失败: {e}")
                features["audio"] = None
        else:
            features["audio"] = None
        
        # 生理特征
        features["health"] = feature_extractor.health_extractor.extract_from_dict(health_data)
        
        # 用药特征
        features["medication"] = feature_extractor.medication_extractor.extract_from_dict(medication_data)
        
        # 预测
        result = predictor.predict(
            video_features=features.get("video"),
            audio_features=features.get("audio"),
            health_features=features.get("health"),
            medication_features=features.get("medication")
        )
    
    # 显示结果
    st.markdown("### 📊 评估结果")
    
    # 风险等级卡片
    risk_class = f"risk-{result['risk_name']}"
    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h2>风险等级: {result['risk_name_cn']}</h2>
        <p style="font-size: 1.2rem;">置信度: {result['confidence']:.1%}</p>
        <p style="font-size: 1rem;">{result['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 概率分布图
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 概率分布柱状图
        fig = go.Figure(data=[
            go.Bar(
                x=['低风险', '中风险', '高风险'],
                y=[result['probabilities']['low'], result['probabilities']['medium'], result['probabilities']['high']],
                marker_color=['#4CAF50', '#FF9800', '#F44336']
            )
        ])
        fig.update_layout(
            title="风险概率分布",
            yaxis_title="概率",
            yaxis_range=[0, 1],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 详细信息
        st.markdown("#### 详细信息")
        st.write(f"**处理建议**: {result['action']}")
        
        if result['missing_modalities']:
            st.warning(f"**缺失模态**: {', '.join(result['missing_modalities'])}")
        else:
            st.success("**数据完整**: 所有模态数据均已提供")

    # ---- 模态贡献分析（可解释性中间结果）----
    st.markdown("### 🔬 模态贡献分析")

    modality_weights = result.get("modality_weights", {})
    attn_matrix = result.get("attention_matrix")
    modality_names = result.get("modality_names", ["视频", "音频", "生理", "用药"])

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 模态全局权重")
        st.caption("模型学习到的各模态对最终决策的静态贡献（softmax 归一化）")
        if modality_weights:
            names = list(modality_weights.keys())
            values = list(modality_weights.values())
            fig_w = go.Figure(data=[
                go.Bar(
                    x=names,
                    y=values,
                    marker_color=["#42A5F5", "#66BB6A", "#FFCA28", "#EF5350"],
                    text=[f"{v:.1%}" for v in values],
                    textposition="outside",
                )
            ])
            fig_w.update_layout(
                yaxis_title="权重",
                yaxis_range=[0, 1],
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("无法获取模态权重")

    with col_b:
        st.markdown("#### 跨模态注意力热力图")
        st.caption("每行=query 模态，每列=key 模态；值越高表示 query 越关注该 key")
        if attn_matrix:
            import numpy as _np
            mat = _np.array(attn_matrix)
            fig_h = go.Figure(data=go.Heatmap(
                z=mat,
                x=modality_names,
                y=modality_names,
                colorscale="YlOrRd",
                text=[[f"{v:.2f}" for v in row] for row in mat],
                texttemplate="%{text}",
                hovertemplate="Q=%{y} K=%{x} attn=%{z:.3f}<extra></extra>",
            ))
            fig_h.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Key 模态",
                yaxis_title="Query 模态",
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("注意力矩阵不可用（可能模型未返回）")


def show_weekly_report():
    """显示健康周报页面"""
    from inference.weekly_report import (
        generate_weekly_records, analyze_records,
        generate_report_text, save_report_json, RISK_NAMES, RISK_COLORS,
    )

    st.markdown('<p class="main-header">📅 健康周报</p>', unsafe_allow_html=True)
    st.markdown("基于多模态模型对一周内每日数据的真实预测，自动生成结构化周报。")

    # 控制栏
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days = st.number_input("周报天数", min_value=3, max_value=14, value=7)
    with col2:
        seed = st.number_input("随机种子", min_value=0, max_value=999, value=42)
    with col3:
        st.write("")
        generate_btn = st.button("生成周报", type="primary")

    if generate_btn:
        predictor = load_predictor()
        feature_extractor = load_feature_extractor()
        if predictor is None:
            st.error("模型未加载，无法生成周报")
            return

        with st.spinner("正在生成每日评估记录..."):
            records = generate_weekly_records(
                predictor, feature_extractor, days=int(days), seed=int(seed)
            )
            analysis = analyze_records(records)
            report_text = generate_report_text(records, analysis)

        if "error" in analysis:
            st.error("分析失败: " + str(analysis["error"]))
            return

        # ---- 摘要卡片 ----
        st.markdown("### 本周概览")
        rd = analysis.get("risk_distribution", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("低风险天数", f"{rd.get(0,0)} 天")
        with m2:
            st.metric("中风险天数", f"{rd.get(1,0)} 天")
        with m3:
            st.metric("高风险天数", f"{rd.get(2,0)} 天")
        with m4:
            st.metric("平均置信度", f"{analysis.get('avg_confidence',0):.0%}")

        # ---- 风险趋势折线图 ----
        st.markdown("### 每日风险趋势")
        trend = analysis.get("risk_trend", [])
        dates = analysis.get("dates", [])
        if trend and dates:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=dates, y=trend, mode="lines+markers",
                name="风险等级",
                line=dict(color="#FF6F00", width=2),
                marker=dict(size=8),
                text=[RISK_NAMES.get(t, "?") for t in trend],
                hovertemplate="%{x}<br>风险: %{text}<extra></extra>",
            ))
            fig_t.update_layout(
                yaxis=dict(
                    tickvals=[0, 1, 2],
                    ticktext=["低风险", "中风险", "高风险"],
                    range=[-0.3, 2.3],
                ),
                yaxis_title="风险等级",
                xaxis_title="日期",
                height=300,
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # ---- 健康指标周均值 ----
        ha = analysis.get("health_avg", {})
        if ha:
            st.markdown("### 健康指标周均值")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("心率", f"{ha.get('heart_rate',0):.0f}", "次/分")
            with c2:
                st.metric("血氧", f"{ha.get('blood_oxygen',0):.0f}%")
            with c3:
                st.metric("收缩压", f"{ha.get('systolic',0):.0f}")
            with c4:
                st.metric("舒张压", f"{ha.get('diastolic',0):.0f}")
            with c5:
                st.metric("日均步数", f"{ha.get('steps',0):.0f}")

        # ---- 模态贡献 ----
        mw = analysis.get("modality_weight_avg", {})
        if mw:
            st.markdown("### 模态贡献（模型平均权重）")
            fig_mw = go.Figure(data=[
                go.Bar(
                    x=list(mw.keys()), y=list(mw.values()),
                    marker_color=["#42A5F5", "#66BB6A", "#FFCA28", "#EF5350"],
                    text=[f"{v:.1%}" for v in mw.values()],
                    textposition="outside",
                )
            ])
            fig_mw.update_layout(yaxis_range=[0, 1], height=250)
            st.plotly_chart(fig_mw, use_container_width=True)

        # ---- 风险提醒 ----
        alerts = analysis.get("alerts", [])
        if alerts:
            st.markdown("### 风险提醒")
            for a in alerts:
                st.error(a)

        # ---- 照护建议 ----
        suggestions = analysis.get("suggestions", [])
        if suggestions:
            st.markdown("### 照护建议")
            for s in suggestions:
                st.info(s)

        # ---- 每日明细表格 ----
        st.markdown("### 每日评估明细")
        table_data = []
        for r in records:
            if r.get("risk_level", -1) >= 0:
                table_data.append({
                    "日期": r["date"],
                    "风险等级": r["risk_name"],
                    "置信度": f"{r['confidence']:.0%}",
                    "场景": r.get("scenario", ""),
                    "缺失模态": ", ".join(r.get("missing_modalities", [])) or "无",
                })
            else:
                table_data.append({
                    "日期": r["date"],
                    "风险等级": "评估失败",
                    "置信度": "-",
                    "场景": r.get("scenario", ""),
                    "缺失模态": "-",
                })
        st.dataframe(table_data, use_container_width=True)

        # ---- 文本报报告 + 导出 ----
        st.markdown("### 完整周报文本")
        st.text(report_text)

        # 保存到文件
        report_path = os.path.join("reports", f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        save_report_json(records, analysis, report_text, report_path)
        st.success(f"周报已保存至: {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            st.download_button(
                "下载周报 (JSON)",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="application/json",
            )


def show_health_data():
    """显示健康数据页面"""
    st.markdown('<p class="main-header">❤️ 健康数据管理</p>', unsafe_allow_html=True)
    
    st.markdown("### 生理数据输入")
    
    # 输入表单
    with st.form("health_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            heart_rate = st.number_input("心率 (bpm)", 30, 200, 75)
            blood_oxygen = st.number_input("血氧 (%)", 70, 100, 97)
        
        with col2:
            systolic = st.number_input("收缩压 (mmHg)", 60, 250, 120)
            diastolic = st.number_input("舒张压 (mmHg)", 40, 150, 80)
        
        steps = st.number_input("步数", 0, 50000, 2000)
        sleep_hours = st.number_input("睡眠时长 (小时)", 0, 24, 7)
        
        submitted = st.form_submit_button("保存数据")
    
    if submitted:
        st.success("✅ 数据已保存")
    
    # 显示趋势图（模拟数据）
    st.markdown("### 历史趋势")
    
    # 生成模拟历史数据
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('心率趋势', '血氧趋势', '血压趋势', '步数趋势')
    )
    
    # 心率
    fig.add_trace(
        go.Scatter(x=dates, y=np.random.normal(75, 5, 7), name='心率'),
        row=1, col=1
    )
    
    # 血氧
    fig.add_trace(
        go.Scatter(x=dates, y=np.random.normal(97, 1, 7), name='血氧'),
        row=1, col=2
    )
    
    # 血压
    fig.add_trace(
        go.Scatter(x=dates, y=np.random.normal(120, 10, 7), name='收缩压'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=np.random.normal(80, 5, 7), name='舒张压'),
        row=2, col=1
    )
    
    # 步数
    fig.add_trace(
        go.Bar(x=dates, y=np.random.randint(1000, 5000, 7), name='步数'),
        row=2, col=2
    )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def show_medication():
    """显示用药管理页面"""
    st.markdown('<p class="main-header">💊 用药管理</p>', unsafe_allow_html=True)
    
    # 今日用药计划
    st.markdown("### 今日用药计划")
    
    medications = [
        {"name": "降压药", "time": "08:00", "dosage": "1片", "taken": True},
        {"name": "降压药", "time": "20:00", "dosage": "1片", "taken": False},
        {"name": "降糖药", "time": "07:30", "dosage": "1片", "taken": True},
        {"name": "钙片", "time": "12:00", "dosage": "2片", "taken": True},
    ]
    
    df = pd.DataFrame(medications)
    df["状态"] = df["taken"].apply(lambda x: "✅ 已服用" if x else "⏳ 待服用")
    
    st.dataframe(
        df[["name", "time", "dosage", "状态"]],
        column_config={
            "name": "药品名称",
            "time": "服药时间",
            "dosage": "剂量",
        },
        use_container_width=True
    )
    
    # 用药统计
    st.markdown("### 用药统计")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("今日应服", "4 次")
    with col2:
        st.metric("已服用", "3 次")
    with col3:
        st.metric("依从率", "75%")


def show_system_info():
    """显示系统说明"""
    st.markdown('<p class="main-header">ℹ️ 系统说明</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 关于智护家
    
    **智护家**是一个面向独居老人的多模态居家健康监护系统，融合视频、音频、生理数据和用药信息，实现综合风险评估。
    
    ### 系统特点
    
    1. **多模态融合**: 综合视频、音频、生理、用药四种模态数据
    2. **智能预测**: 基于深度学习的风险预测模型
    3. **缺失容忍**: 即使部分数据缺失，系统仍能给出合理预测
    4. **实时反馈**: 快速响应，即时给出风险评估结果
    
    ### 风险等级说明
    
    | 等级 | 说明 | 处理建议 |
    |------|------|---------|
    | 低风险 | 状态正常 | 记录到系统，持续观察 |
    | 中风险 | 存在潜在风险 | 推送通知，建议关注 |
    | 高风险 | 需要立即关注 | 立即报警，联系确认 |
    
    ### 使用说明
    
    1. 在"风险评估"页面上传数据（所有数据均为可选）
    2. 点击"开始风险评估"按钮
    3. 查看风险等级和详细分析
    
    ### 技术架构
    
    - 视频编码器: 3D CNN + Transformer
    - 音频编码器: CNN + Transformer
    - 生理编码器: Transformer
    - 融合网络: Cross-Modal Attention
    """)


def main():
    """主函数"""
    page = show_sidebar()
    
    if "风险评估" in page:
        show_risk_assessment()
    elif "健康周报" in page:
        show_weekly_report()
    elif "健康数据" in page:
        show_health_data()
    elif "用药管理" in page:
        show_medication()
    elif "系统说明" in page:
        show_system_info()


if __name__ == "__main__":
    main()
