"""
智护家 - 智慧居家养老多模态监护系统
Streamlit 演示界面
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import SYSTEM_CONFIG, VIDEO_CONFIG, AUDIO_CONFIG, HEALTH_CONFIG, FUSION_CONFIG
from modules.video.fall_detector import FallDetector, process_video
from modules.audio.audio_detector import AudioDetector
from modules.health.health_monitor import HealthMonitor, generate_sample_health_data
from modules.medication.medication_manager import MedicationManager, create_sample_medication_plan
from modules.fusion.fusion_engine import FusionEngine, RiskLevel
from modules.report.weekly_report import ReportGenerator


# 页面配置
st.set_page_config(
    page_title="智护家 - 智慧居家养老多模态监护系统",
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
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'health_monitor' not in st.session_state:
        st.session_state.health_monitor = HealthMonitor()
        st.session_state.health_data = generate_sample_health_data(7)
        st.session_state.health_monitor.set_baseline(st.session_state.health_data)
    
    if 'medication_manager' not in st.session_state:
        st.session_state.medication_manager = create_sample_medication_plan()
    
    if 'fusion_engine' not in st.session_state:
        st.session_state.fusion_engine = FusionEngine()
    
    if 'safety_events' not in st.session_state:
        st.session_state.safety_events = []
    
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator()


def show_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/home.png", width=80)
        st.title("智护家")
        st.caption("智慧居家养老多模态监护系统")
        
        st.divider()
        
        # 导航菜单
        page = st.radio(
            "功能导航",
            ["🏠 系统概览", "📹 视频监测", "🔊 音频监测", "❤️ 健康监测", 
             "💊 用药管理", "🔗 多模态融合", "📊 健康周报"],
            key="navigation"
        )
        
        st.divider()
        
        # 系统状态
        st.subheader("系统状态")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("监测天数", "7")
        with col2:
            st.metric("数据完整率", "95%")
        
        # 快速报警状态
        st.subheader("报警状态")
        if st.session_state.safety_events:
            high_risk = sum(1 for e in st.session_state.safety_events if e.get('risk_level') == 'high')
            if high_risk > 0:
                st.error(f"⚠️ {high_risk} 个高风险事件")
            else:
                st.success("✅ 无高风险事件")
        else:
            st.success("✅ 系统正常")
        
        return page


def show_overview():
    """显示系统概览页面"""
    st.markdown('<p class="main-header">🏠 智护家 - 系统概览</p>', unsafe_allow_html=True)
    
    # 系统介绍
    st.markdown("""
    ### 系统简介
    **智护家**是一个面向独居老人的多模态居家健康监护与陪护建议系统。
    系统融合视频、音频、生理数据和用药信息，实现：
    - 🎯 **安全事件识别**：跌倒检测、异常行为识别
    - ❤️ **健康状态监测**：心率、血氧、血压异常预警
    - 💊 **用药管理**：服药提醒、漏服检测
    - 📊 **健康周报**：自动生成健康报告和陪护建议
    """)
    
    st.divider()
    
    # 核心指标
    st.subheader("核心监测指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("今日步数", "2,450", "↑ 12%")
    with col2:
        st.metric("平均心率", "72 bpm", "正常")
    with col3:
        st.metric("血氧饱和度", "97%", "正常")
    with col4:
        st.metric("用药依从率", "85%", "↓ 5%")
    
    # 系统架构图
    st.subheader("系统架构")
    st.markdown("""
    ```
    数据采集层 → 数据预处理层 → 单模态分析层 → 多模态融合层 → 应用服务层
         │              │              │              │              │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │ 摄像头  │   │ 视频抽帧 │   │ 跌倒检测 │   │ 风险评估 │   │ 实时报警 │
    │ 麦克风  │   │ 音频降噪 │   │ 音频识别 │   │ 异常判断 │   │ 健康周报 │
    │ 手环    │   │ 数据清洗 │   │ 健康监测 │   │ 等级划分 │   │ 陪护建议 │
    │ 用药计划│   │ 结构化   │   │ 用药管理 │   │          │   │          │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
    ```
    """)
    
    # 最近事件
    st.subheader("最近事件")
    if st.session_state.safety_events:
        for event in st.session_state.safety_events[-5:]:
            risk_class = f"risk-{event.get('risk_level', 'low')}"
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <strong>{event.get('type', '未知事件')}</strong> - 
                {event.get('timestamp', '')[:19]}<br>
                {event.get('description', '')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无事件记录")


def show_video_monitoring():
    """显示视频监测页面"""
    st.markdown('<p class="main-header">📹 视频监测</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 跌倒检测功能
    使用 MediaPipe Pose 进行人体姿态估计，结合规则判断检测跌倒事件。
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("视频输入")
        # 视频上传
        uploaded_file = st.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            if st.button("开始分析", key="analyze_video"):
                with st.spinner("正在分析视频..."):
                    # 这里可以调用实际的视频分析
                    st.success("分析完成！")
                    
                    # 模拟结果
                    st.session_state.safety_events.append({
                        'type': 'suspected_fall',
                        'risk_level': 'medium',
                        'timestamp': datetime.now().isoformat(),
                        'description': '检测到疑似跌倒行为'
                    })
        
        # 摄像头实时监测
        st.subheader("实时摄像头")
        if st.button("开启摄像头"):
            st.info("摄像头功能需要实际硬件支持")
    
    with col2:
        st.subheader("检测参数")
        fall_threshold = st.slider("跌倒高度阈值", 0.1, 0.5, 0.3)
        stillness_threshold = st.slider("静止时间阈值(秒)", 30, 120, 60)
        
        st.subheader("检测状态")
        st.metric("姿态估计", "就绪")
        st.metric("跌倒检测", "就绪")
        
        st.subheader("检测说明")
        st.markdown("""
        **跌倒判断依据：**
        1. 人体重心快速下降
        2. 躯干角度接近水平
        3. 跌倒后长时间不动
        """)


def show_audio_monitoring():
    """显示音频监测页面"""
    st.markdown('<p class="main-header">🔊 音频监测</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 音频异常检测
    检测呼救声、撞击声、摔倒声等异常音频事件。
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("音频输入")
        uploaded_audio = st.file_uploader("上传音频文件", type=['wav', 'mp3', 'ogg'])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            
            if st.button("分析音频", key="analyze_audio"):
                with st.spinner("正在分析音频..."):
                    # 创建音频检测器
                    detector = AudioDetector()
                    
                    # 模拟分析结果
                    st.success("分析完成！")
                    
                    # 显示结果
                    st.subheader("分析结果")
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        st.metric("撞击声", "未检测到")
                    with result_col2:
                        st.metric("呼救声", "未检测到")
    
    with col2:
        st.subheader("检测类型")
        detect_types = st.multiselect(
            "选择检测类型",
            ["撞击声", "呼救声", "尖叫声", "咳嗽声", "异常静默"],
            default=["撞击声", "呼救声"]
        )
        
        st.subheader("关键词列表")
        keywords = st.text_area(
            "呼救关键词",
            "救命, 帮帮我, 我摔倒了, 难受, 过来一下",
            height=100
        )
        
        st.subheader("检测说明")
        st.markdown("""
        **音频特征提取：**
        - Mel频谱图
        - MFCC特征
        - 能量特征
        - 频谱质心
        """)


def show_health_monitoring():
    """显示健康监测页面"""
    st.markdown('<p class="main-header">❤️ 健康监测</p>', unsafe_allow_html=True)
    
    # 获取健康数据
    health_data = st.session_state.health_data
    monitor = st.session_state.health_monitor
    
    # 实时指标
    st.subheader("实时健康指标")
    col1, col2, col3, col4 = st.columns(4)
    
    latest = health_data.iloc[-1] if len(health_data) > 0 else {}
    
    with col1:
        st.metric(
            "心率", 
            f"{latest.get('heart_rate', 0):.0f} bpm",
            delta="正常" if 60 <= latest.get('heart_rate', 0) <= 100 else "异常"
        )
    with col2:
        st.metric(
            "血氧", 
            f"{latest.get('blood_oxygen', 0):.0f}%",
            delta="正常" if latest.get('blood_oxygen', 0) >= 95 else "偏低"
        )
    with col3:
        st.metric(
            "收缩压", 
            f"{latest.get('systolic', 0):.0f} mmHg",
            delta="正常" if 90 <= latest.get('systolic', 0) <= 140 else "异常"
        )
    with col4:
        st.metric(
            "舒张压", 
            f"{latest.get('diastolic', 0):.0f} mmHg",
            delta="正常" if 60 <= latest.get('diastolic', 0) <= 90 else "异常"
        )
    
    # 趋势图表
    st.subheader("健康趋势")
    
    tab1, tab2, tab3 = st.tabs(["心率", "血氧", "血压"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=health_data['timestamp'],
            y=health_data['heart_rate'],
            mode='lines',
            name='心率',
            line=dict(color='#E91E63')
        ))
        fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="下限")
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="上限")
        fig.update_layout(title="心率趋势", xaxis_title="时间", yaxis_title="bpm")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=health_data['timestamp'],
            y=health_data['blood_oxygen'],
            mode='lines',
            name='血氧',
            line=dict(color='#2196F3')
        ))
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="下限")
        fig.update_layout(title="血氧趋势", xaxis_title="时间", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=health_data['timestamp'],
            y=health_data['systolic'],
            mode='lines',
            name='收缩压',
            line=dict(color='#FF5722')
        ))
        fig.add_trace(go.Scatter(
            x=health_data['timestamp'],
            y=health_data['diastolic'],
            mode='lines',
            name='舒张压',
            line=dict(color='#FF9800')
        ))
        fig.update_layout(title="血压趋势", xaxis_title="时间", yaxis_title="mmHg")
        st.plotly_chart(fig, use_container_width=True)
    
    # 异常检测
    st.subheader("异常检测")
    
    # 检测最新数据的异常
    test_record = {
        'timestamp': datetime.now().isoformat(),
        'heart_rate': latest.get('heart_rate', 75),
        'blood_oxygen': latest.get('blood_oxygen', 96),
        'systolic': latest.get('systolic', 120),
        'diastolic': latest.get('diastolic', 80),
        'steps': latest.get('steps', 1000),
        'sleep_hours': latest.get('sleep_hours', 7),
    }
    
    anomalies = monitor.detect_anomaly(test_record)
    
    if anomalies:
        for anomaly in anomalies:
            severity_color = {
                'low': '🟡',
                'medium': '🟠',
                'high': '🔴'
            }
            st.warning(f"{severity_color.get(anomaly.severity, '⚪')} {anomaly.description}")
    else:
        st.success("✅ 当前各项指标正常")
    
    # 阈值设置
    st.subheader("阈值设置")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("心率下限", 40, 80, 50)
        st.number_input("心率上限", 100, 160, 120)
    with col2:
        st.number_input("血氧下限", 80, 95, 90)
        st.number_input("收缩压上限", 140, 200, 160)


def show_medication_management():
    """显示用药管理页面"""
    st.markdown('<p class="main-header">💊 用药管理</p>', unsafe_allow_html=True)
    
    manager = st.session_state.medication_manager
    
    # 今日用药计划
    st.subheader("今日用药计划")
    schedule = manager.get_today_schedule()
    
    if schedule:
        df_data = []
        for item in schedule:
            df_data.append({
                '药品名称': item['medication_name'],
                '剂量': item['dosage'],
                '计划时间': item['scheduled_time'].strftime('%H:%M'),
                '状态': '✅ 已服用' if item['status'] == 'taken' else 
                        '❌ 漏服' if item['status'] == 'missed' else '⏳ 待服用',
                '备注': '饭前' if item['before_meal'] else '饭后'
            })
        
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
    else:
        st.info("暂无用药计划")
    
    # 用药依从性
    st.subheader("用药依从性统计")
    stats = manager.get_adherence_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("依从率", f"{stats['adherence_rate']}%")
    with col2:
        st.metric("已服用", stats['taken'])
    with col3:
        st.metric("漏服", stats['missed'])
    
    # 依从性图表
    if stats['by_medication']:
        fig = go.Figure(data=[
            go.Bar(
                x=list(stats['by_medication'].keys()),
                y=[v['adherence_rate'] for v in stats['by_medication'].values()],
                marker_color=['#4CAF50' if v['adherence_rate'] >= 80 else '#FF9800' 
                             for v in stats['by_medication'].values()]
            )
        ])
        fig.update_layout(title="各药品依从率", yaxis_title="依从率(%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # 添加药品
    st.subheader("添加药品")
    col1, col2 = st.columns(2)
    
    with col1:
        med_name = st.text_input("药品名称")
        med_dosage = st.text_input("剂量", value="1片")
        med_frequency = st.number_input("每日次数", 1, 4, 1)
    
    with col2:
        med_times = st.text_input("服药时间(逗号分隔)", value="08:00")
        med_before_meal = st.checkbox("饭前服用")
        med_notes = st.text_area("备注")
    
    if st.button("添加药品"):
        if med_name:
            manager.add_medication(
                name=med_name,
                dosage=med_dosage,
                frequency=med_frequency,
                times=[t.strip() for t in med_times.split(',')],
                before_meal=med_before_meal,
                notes=med_notes
            )
            st.success(f"已添加药品: {med_name}")
    
    # 检查提醒
    st.subheader("用药提醒")
    reminders = manager.check_reminders()
    if reminders:
        for reminder in reminders:
            if reminder['type'] == 'reminder':
                st.info(f"⏰ {reminder['message']}")
            else:
                st.warning(f"⚠️ {reminder['message']}")
    else:
        st.info("暂无用药提醒")


def show_fusion():
    """显示多模态融合页面"""
    st.markdown('<p class="main-header">🔗 多模态融合决策</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 多模态融合策略
    系统采用"单模态识别 + 多模态融合决策"的方式，综合判断风险等级。
    """)
    
    engine = st.session_state.fusion_engine
    
    # 融合权重设置
    st.subheader("融合权重配置")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        video_weight = st.slider("视频权重", 0.0, 1.0, 0.4)
    with col2:
        audio_weight = st.slider("音频权重", 0.0, 1.0, 0.25)
    with col3:
        health_weight = st.slider("健康权重", 0.0, 1.0, 0.25)
    with col4:
        medication_weight = st.slider("用药权重", 0.0, 1.0, 0.1)
    
    # 模拟场景测试
    st.subheader("场景模拟测试")
    
    st.markdown("**模拟跌倒事件：**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        video_conf = st.slider("视频置信度", 0.0, 1.0, 0.85)
        video_risk = st.slider("视频风险分", 0.0, 1.0, 0.8)
    
    with col2:
        audio_conf = st.slider("音频置信度", 0.0, 1.0, 0.7)
        audio_risk = st.slider("音频风险分", 0.0, 1.0, 0.6)
    
    with col3:
        health_conf = st.slider("健康置信度", 0.0, 1.0, 0.6)
        health_risk = st.slider("健康风险分", 0.0, 1.0, 0.5)
    
    if st.button("进行融合评估"):
        # 进行融合评估
        assessment = engine.assess_fall_event(
            video_result={'confidence': video_conf, 'risk_score': video_risk, 'events': []},
            audio_result={'confidence': audio_conf, 'risk_score': audio_risk, 'events': []},
            health_result={'confidence': health_conf, 'risk_score': health_risk, 'events': []}
        )
        
        # 显示结果
        st.subheader("评估结果")
        
        risk_color = {
            RiskLevel.LOW: '#4CAF50',
            RiskLevel.MEDIUM: '#FF9800',
            RiskLevel.HIGH: '#F44336'
        }
        
        st.markdown(f"""
        <div style="background-color: {risk_color[assessment.risk_level]}20; 
                    border-left: 5px solid {risk_color[assessment.risk_level]}; 
                    padding: 1rem; border-radius: 5px;">
            <h3>风险等级: {assessment.risk_level.value.upper()}</h3>
            <p><strong>风险分数:</strong> {assessment.risk_score}</p>
            <p><strong>建议:</strong> {assessment.recommendation}</p>
            <p><strong>需要立即行动:</strong> {'是' if assessment.requires_immediate_action else '否'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 添加到事件记录
        st.session_state.safety_events.append({
            'type': 'simulated_fall',
            'risk_level': assessment.risk_level.value,
            'timestamp': assessment.timestamp,
            'description': f"模拟跌倒测试 - 风险分数: {assessment.risk_score}"
        })
    
    # 风险趋势
    st.subheader("风险趋势")
    trend = engine.get_risk_trend()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("趋势", trend['trend'])
    with col2:
        st.metric("平均风险分", trend['average_score'])
    with col3:
        st.metric("评估次数", trend['assessment_count'])


def show_weekly_report():
    """显示健康周报页面"""
    st.markdown('<p class="main-header">📊 健康周报</p>', unsafe_allow_html=True)
    
    generator = st.session_state.report_generator
    
    # 生成报告
    report = generator.generate_weekly_report(
        health_data=st.session_state.health_data,
        safety_events=st.session_state.safety_events,
        medication_stats=st.session_state.medication_manager.get_adherence_stats()
    )
    
    # 报告概览
    st.subheader("报告概览")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("监测周期", f"{report.monitoring_days}天")
    with col2:
        st.metric("数据完整率", f"{report.data_completeness}%")
    with col3:
        st.metric("安全事件", report.safety_events['total_events'])
    with col4:
        st.metric("高风险事件", report.safety_events['high_risk_events'])
    
    # 生理指标
    st.subheader("生理指标趋势")
    
    if report.health_indicators:
        df_data = []
        for indicator, data in report.health_indicators.items():
            df_data.append({
                '指标': data['name'],
                '平均值': f"{data['mean']}{data['unit']}",
                '范围': f"{data['min']}-{data['max']}",
                '趋势': data['trend'],
                '状态': '正常' if data['in_normal_range'] else '异常'
            })
        
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
    
    # 活动与睡眠
    st.subheader("活动与睡眠")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'steps' in report.activity_summary:
            steps = report.activity_summary['steps']
            st.metric("日均步数", f"{steps['daily_average']:.0f}")
            st.metric("活动水平", steps['level'])
    
    with col2:
        if 'sleep' in report.activity_summary:
            sleep = report.activity_summary['sleep']
            st.metric("平均睡眠", f"{sleep['average_hours']:.1f}小时")
            st.metric("睡眠状态", sleep['status'])
    
    # 安全事件汇总
    st.subheader("安全事件汇总")
    events = report.safety_events
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("跌倒事件", events['fall_events'])
    with col2:
        st.metric("疑似跌倒", events['suspected_fall_events'])
    with col3:
        st.metric("呼救事件", events['help_call_events'])
    with col4:
        st.metric("高风险事件", events['high_risk_events'])
    
    # 陪护建议
    st.subheader("陪护建议")
    for i, suggestion in enumerate(report.care_suggestions, 1):
        st.info(f"💡 {suggestion}")
    
    # 风险提示
    if report.risk_alerts:
        st.subheader("风险提示")
        for alert in report.risk_alerts:
            st.warning(alert)
    
    # 导出报告
    st.subheader("导出报告")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("导出文本报告"):
            text_report = generator.format_report_text(report)
            st.download_button(
                label="下载文本报告",
                data=text_report,
                file_name=f"health_report_{report.report_date}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("导出JSON报告"):
            json_report = generator.export_report_json(report)
            st.download_button(
                label="下载JSON报告",
                data=json_report,
                file_name=f"health_report_{report.report_date}.json",
                mime="application/json"
            )


def main():
    """主函数"""
    # 初始化
    init_session_state()
    
    # 显示侧边栏并获取当前页面
    page = show_sidebar()
    
    # 根据选择显示不同页面
    if "系统概览" in page:
        show_overview()
    elif "视频监测" in page:
        show_video_monitoring()
    elif "音频监测" in page:
        show_audio_monitoring()
    elif "健康监测" in page:
        show_health_monitoring()
    elif "用药管理" in page:
        show_medication_management()
    elif "多模态融合" in page:
        show_fusion()
    elif "健康周报" in page:
        show_weekly_report()


if __name__ == "__main__":
    main()
