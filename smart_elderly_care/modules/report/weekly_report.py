"""
健康周报生成模块
生成健康周报和陪护建议
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import json


@dataclass
class WeeklyReport:
    """健康周报数据结构"""
    report_date: str
    period_start: str
    period_end: str
    monitoring_days: int
    data_completeness: float
    
    # 生理指标
    health_indicators: Dict = field(default_factory=dict)
    
    # 活动与睡眠
    activity_summary: Dict = field(default_factory=dict)
    
    # 安全事件
    safety_events: Dict = field(default_factory=dict)
    
    # 用药情况
    medication_summary: Dict = field(default_factory=dict)
    
    # 陪护建议
    care_suggestions: List[str] = field(default_factory=list)
    
    # 风险提示
    risk_alerts: List[str] = field(default_factory=list)


class ReportGenerator:
    """健康周报生成器"""
    
    def __init__(self):
        self.report_templates = {
            "health_trend": {
                "increasing": "{indicator}呈上升趋势，当前平均值{mean}，建议持续观察。",
                "decreasing": "{indicator}呈下降趋势，当前平均值{mean}，建议关注。",
                "stable": "{indicator}保持稳定，当前平均值{mean}。",
            },
            "activity": {
                "low": "本周活动量较低，建议适当增加户外活动。",
                "normal": "本周活动量正常。",
                "high": "本周活动量较高，注意休息。",
            },
            "sleep": {
                "insufficient": "本周睡眠时长不足，建议调整作息。",
                "excessive": "本周睡眠时间过长，建议关注精神状态。",
                "normal": "本周睡眠时长正常。",
            },
            "medication": {
                "good": "本周用药依从性良好，继续保持。",
                "needs_attention": "本周有{missed}次漏服药，建议加强提醒。",
                "poor": "本周用药依从性较差，建议家属协助管理。",
            }
        }
    
    def generate_weekly_report(self, 
                               health_data: pd.DataFrame,
                               safety_events: List[Dict],
                               medication_stats: Dict,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> WeeklyReport:
        """生成健康周报"""
        
        # 确定报告周期
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now()
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = end_dt - timedelta(days=7)
        
        # 筛选时间范围内的数据
        if 'timestamp' in health_data.columns:
            health_data['timestamp'] = pd.to_datetime(health_data['timestamp'])
            period_data = health_data[
                (health_data['timestamp'] >= start_dt) & 
                (health_data['timestamp'] <= end_dt)
            ]
        else:
            period_data = health_data
        
        monitoring_days = len(period_data['timestamp'].dt.date.unique()) if len(period_data) > 0 else 0
        
        # 创建报告
        report = WeeklyReport(
            report_date=datetime.now().strftime("%Y-%m-%d"),
            period_start=start_dt.strftime("%Y-%m-%d"),
            period_end=end_dt.strftime("%Y-%m-%d"),
            monitoring_days=monitoring_days,
            data_completeness=self._calculate_completeness(period_data),
        )
        
        # 分析生理指标
        report.health_indicators = self._analyze_health_indicators(period_data)
        
        # 分析活动与睡眠
        report.activity_summary = self._analyze_activity(period_data)
        
        # 汇总安全事件
        report.safety_events = self._summarize_safety_events(safety_events)
        
        # 汇总用药情况
        report.medication_summary = medication_stats
        
        # 生成陪护建议
        report.care_suggestions = self._generate_care_suggestions(report)
        
        # 生成风险提示
        report.risk_alerts = self._generate_risk_alerts(report)
        
        return report
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """计算数据完整率"""
        if len(data) == 0:
            return 0.0
        
        # 检查关键字段的完整率
        key_columns = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic']
        available_cols = [col for col in key_columns if col in data.columns]
        
        if not available_cols:
            return 0.0
        
        completeness = data[available_cols].notna().mean().mean()
        return round(completeness * 100, 1)
    
    def _analyze_health_indicators(self, data: pd.DataFrame) -> Dict:
        """分析生理指标"""
        indicators = {}
        
        indicator_config = {
            'heart_rate': {'name': '心率', 'unit': 'bpm', 'normal_range': (60, 100)},
            'blood_oxygen': {'name': '血氧', 'unit': '%', 'normal_range': (95, 100)},
            'systolic': {'name': '收缩压', 'unit': 'mmHg', 'normal_range': (90, 140)},
            'diastolic': {'name': '舒张压', 'unit': 'mmHg', 'normal_range': (60, 90)},
        }
        
        for col, config in indicator_config.items():
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            mean_val = values.mean()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            
            # 判断趋势
            if len(values) >= 6:
                first_half = values[:len(values)//2].mean()
                second_half = values[len(values)//2:].mean()
                change_ratio = (second_half - first_half) / (first_half + 1e-6)
                
                if change_ratio > 0.1:
                    trend = "increasing"
                elif change_ratio < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # 判断是否在正常范围
            normal_range = config['normal_range']
            in_range = normal_range[0] <= mean_val <= normal_range[1]
            
            indicators[col] = {
                'name': config['name'],
                'unit': config['unit'],
                'mean': round(mean_val, 1),
                'std': round(std_val, 1),
                'min': round(min_val, 1),
                'max': round(max_val, 1),
                'trend': trend,
                'in_normal_range': in_range,
                'normal_range': normal_range,
            }
        
        return indicators
    
    def _analyze_activity(self, data: pd.DataFrame) -> Dict:
        """分析活动与睡眠"""
        summary = {}
        
        # 步数分析
        if 'steps' in data.columns:
            daily_steps = data.groupby(data['timestamp'].dt.date)['steps'].sum() if 'timestamp' in data.columns else pd.Series()
            
            if len(daily_steps) > 0:
                avg_steps = daily_steps.mean()
                
                if avg_steps < 1000:
                    activity_level = "low"
                elif avg_steps < 5000:
                    activity_level = "normal"
                else:
                    activity_level = "high"
                
                summary['steps'] = {
                    'daily_average': round(avg_steps, 0),
                    'total': int(daily_steps.sum()),
                    'level': activity_level,
                    'days_tracked': len(daily_steps),
                }
        
        # 睡眠分析
        if 'sleep_hours' in data.columns:
            sleep_values = data['sleep_hours'].dropna()
            
            if len(sleep_values) > 0:
                avg_sleep = sleep_values.mean()
                
                if avg_sleep < 6:
                    sleep_status = "insufficient"
                elif avg_sleep > 10:
                    sleep_status = "excessive"
                else:
                    sleep_status = "normal"
                
                summary['sleep'] = {
                    'average_hours': round(avg_sleep, 1),
                    'min_hours': round(sleep_values.min(), 1),
                    'max_hours': round(sleep_values.max(), 1),
                    'status': sleep_status,
                }
        
        return summary
    
    def _summarize_safety_events(self, events: List[Dict]) -> Dict:
        """汇总安全事件"""
        summary = {
            'total_events': len(events),
            'fall_events': 0,
            'suspected_fall_events': 0,
            'help_call_events': 0,
            'long_stillness_events': 0,
            'health_anomaly_events': 0,
            'high_risk_events': 0,
            'event_details': [],
        }
        
        for event in events:
            event_type = event.get('type', 'unknown')
            risk_level = event.get('risk_level', 'low')
            
            if event_type == 'fall':
                summary['fall_events'] += 1
            elif event_type == 'suspected_fall':
                summary['suspected_fall_events'] += 1
            elif event_type == 'help_call':
                summary['help_call_events'] += 1
            elif event_type == 'long_stillness':
                summary['long_stillness_events'] += 1
            elif event_type == 'health_anomaly':
                summary['health_anomaly_events'] += 1
            
            if risk_level == 'high':
                summary['high_risk_events'] += 1
            
            summary['event_details'].append({
                'type': event_type,
                'timestamp': event.get('timestamp', ''),
                'risk_level': risk_level,
                'description': event.get('description', ''),
            })
        
        return summary
    
    def _generate_care_suggestions(self, report: WeeklyReport) -> List[str]:
        """生成陪护建议"""
        suggestions = []
        
        # 基于生理指标的建议
        for indicator, data in report.health_indicators.items():
            if not data.get('in_normal_range', True):
                suggestions.append(
                    f"{data['name']}平均值{data['mean']}{data['unit']}，"
                    f"超出正常范围{data['normal_range']}，建议关注。"
                )
            
            if data.get('trend') == 'increasing' and indicator in ['heart_rate', 'systolic', 'diastolic']:
                suggestions.append(f"{data['name']}呈上升趋势，建议持续监测。")
        
        # 基于活动的建议
        if 'steps' in report.activity_summary:
            steps_data = report.activity_summary['steps']
            if steps_data['level'] == 'low':
                suggestions.append(
                    f"本周日均步数{steps_data['daily_average']:.0f}步，"
                    f"活动量较低，建议适当增加白天陪伴和轻度运动。"
                )
        
        # 基于睡眠的建议
        if 'sleep' in report.activity_summary:
            sleep_data = report.activity_summary['sleep']
            if sleep_data['status'] == 'insufficient':
                suggestions.append(
                    f"本周平均睡眠{sleep_data['average_hours']:.1f}小时，"
                    f"睡眠不足，建议调整作息时间。"
                )
            elif sleep_data['status'] == 'excessive':
                suggestions.append(
                    f"本周平均睡眠{sleep_data['average_hours']:.1f}小时，"
                    f"睡眠时间较长，建议关注精神状态。"
                )
        
        # 基于安全事件的建议
        if report.safety_events['high_risk_events'] > 0:
            suggestions.append(
                f"本周发生{report.safety_events['high_risk_events']}次高风险事件，"
                f"建议加强看护。"
            )
        
        # 基于用药的建议
        if report.medication_summary:
            adherence = report.medication_summary.get('adherence_rate', 100)
            missed = report.medication_summary.get('missed', 0)
            
            if adherence < 80:
                suggestions.append(
                    f"本周用药依从率{adherence}%，有{missed}次漏服，"
                    f"建议设置更明显的提醒或使用智能药盒。"
                )
        
        # 默认建议
        if not suggestions:
            suggestions.append("本周老人整体状态良好，继续保持当前照护方式。")
        
        return suggestions
    
    def _generate_risk_alerts(self, report: WeeklyReport) -> List[str]:
        """生成风险提示"""
        alerts = []
        
        # 高风险事件提示
        if report.safety_events['high_risk_events'] > 0:
            alerts.append(
                f"⚠️ 本周发生{report.safety_events['high_risk_events']}次高风险事件，"
                f"请家属重点关注。"
            )
        
        # 跌倒风险提示
        total_fall_related = (
            report.safety_events['fall_events'] + 
            report.safety_events['suspected_fall_events']
        )
        if total_fall_related > 0:
            alerts.append(
                f"⚠️ 本周检测到{total_fall_related}次跌倒相关事件，"
                f"建议排查居家安全隐患。"
            )
        
        # 健康指标异常提示
        for indicator, data in report.health_indicators.items():
            if data.get('trend') in ['increasing', 'decreasing']:
                if indicator in ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic']:
                    alerts.append(
                        f"⚠️ {data['name']}出现明显{data['trend']}趋势，"
                        f"建议咨询医生。"
                    )
        
        return alerts
    
    def format_report_text(self, report: WeeklyReport) -> str:
        """将报告格式化为文本"""
        lines = []
        
        # 标题
        lines.append("=" * 50)
        lines.append("智护家 - 健康周报")
        lines.append("=" * 50)
        lines.append("")
        
        # 基本信息
        lines.append(f"报告日期: {report.report_date}")
        lines.append(f"监测周期: {report.period_start} 至 {report.period_end}")
        lines.append(f"监测天数: {report.monitoring_days} 天")
        lines.append(f"数据完整率: {report.data_completeness}%")
        lines.append("")
        
        # 生理指标
        lines.append("-" * 50)
        lines.append("【生理指标趋势】")
        lines.append("-" * 50)
        for indicator, data in report.health_indicators.items():
            status = "正常" if data.get('in_normal_range', True) else "异常"
            lines.append(
                f"  {data['name']}: 平均 {data['mean']}{data['unit']} "
                f"(范围: {data['min']}-{data['max']}), {status}"
            )
        lines.append("")
        
        # 活动与睡眠
        lines.append("-" * 50)
        lines.append("【活动与睡眠】")
        lines.append("-" * 50)
        if 'steps' in report.activity_summary:
            steps = report.activity_summary['steps']
            lines.append(f"  日均步数: {steps['daily_average']:.0f} 步")
        if 'sleep' in report.activity_summary:
            sleep = report.activity_summary['sleep']
            lines.append(f"  平均睡眠: {sleep['average_hours']:.1f} 小时")
        lines.append("")
        
        # 安全事件
        lines.append("-" * 50)
        lines.append("【安全事件】")
        lines.append("-" * 50)
        events = report.safety_events
        lines.append(f"  总事件数: {events['total_events']}")
        lines.append(f"  跌倒事件: {events['fall_events']}")
        lines.append(f"  疑似跌倒: {events['suspected_fall_events']}")
        lines.append(f"  呼救事件: {events['help_call_events']}")
        lines.append(f"  高风险事件: {events['high_risk_events']}")
        lines.append("")
        
        # 用药情况
        lines.append("-" * 50)
        lines.append("【用药情况】")
        lines.append("-" * 50)
        if report.medication_summary:
            med = report.medication_summary
            lines.append(f"  用药依从率: {med.get('adherence_rate', 'N/A')}%")
            lines.append(f"  漏服次数: {med.get('missed', 0)}")
        else:
            lines.append("  暂无用药数据")
        lines.append("")
        
        # 陪护建议
        lines.append("-" * 50)
        lines.append("【陪护建议】")
        lines.append("-" * 50)
        for i, suggestion in enumerate(report.care_suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        lines.append("")
        
        # 风险提示
        if report.risk_alerts:
            lines.append("-" * 50)
            lines.append("【风险提示】")
            lines.append("-" * 50)
            for alert in report.risk_alerts:
                lines.append(f"  {alert}")
            lines.append("")
        
        lines.append("=" * 50)
        lines.append("智护家 - 关爱老人，智慧守护")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def export_report_json(self, report: WeeklyReport) -> str:
        """导出报告为JSON格式"""
        return json.dumps({
            'report_date': report.report_date,
            'period': {
                'start': report.period_start,
                'end': report.period_end,
            },
            'monitoring_days': report.monitoring_days,
            'data_completeness': report.data_completeness,
            'health_indicators': report.health_indicators,
            'activity_summary': report.activity_summary,
            'safety_events': report.safety_events,
            'medication_summary': report.medication_summary,
            'care_suggestions': report.care_suggestions,
            'risk_alerts': report.risk_alerts,
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试代码
    print("健康周报生成模块测试")
    generator = ReportGenerator()
    
    # 创建模拟数据
    from modules.health.health_monitor import generate_sample_health_data
    health_data = generate_sample_health_data(7)
    
    # 模拟安全事件
    safety_events = [
        {'type': 'suspected_fall', 'risk_level': 'medium', 'timestamp': '2024-01-15T10:30:00'},
        {'type': 'help_call', 'risk_level': 'high', 'timestamp': '2024-01-16T14:00:00'},
    ]
    
    # 模拟用药统计
    medication_stats = {
        'adherence_rate': 85.7,
        'taken': 12,
        'missed': 2,
    }
    
    # 生成报告
    report = generator.generate_weekly_report(
        health_data=health_data,
        safety_events=safety_events,
        medication_stats=medication_stats
    )
    
    # 输出文本报告
    print(generator.format_report_text(report))
