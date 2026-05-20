"""
生理数据异常检测模块
检测心率、血氧、血压等生理指标异常
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HealthAnomaly:
    """健康异常事件"""
    timestamp: str
    indicator: str
    value: float
    threshold: Tuple[float, float]
    severity: str  # "low", "medium", "high"
    description: str
    confidence: float


class HealthMonitor:
    """生理数据监测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            "heart_rate": {
                "low_threshold": 50,
                "high_threshold": 120,
            },
            "blood_oxygen": {
                "low_threshold": 90,
            },
            "blood_pressure": {
                "systolic_low": 90,
                "systolic_high": 160,
                "diastolic_low": 60,
                "diastolic_high": 100,
            },
            "activity": {
                "low_steps_threshold": 500,
            },
            "sleep": {
                "min_hours": 4,
                "max_hours": 12,
            }
        }
        
        # 个性化基线
        self.baseline = {}
        self.baseline_history = []
        
        # 异常检测模型
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # 历史数据
        self.health_history = []
    
    def set_baseline(self, historical_data: pd.DataFrame):
        """根据历史数据建立个性化基线"""
        if historical_data.empty:
            return
        
        # 计算各指标的统计基线
        indicators = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic', 'steps', 'sleep_hours']
        
        for indicator in indicators:
            if indicator in historical_data.columns:
                values = historical_data[indicator].dropna()
                if len(values) > 0:
                    self.baseline[indicator] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                    }
        
        # 训练Isolation Forest模型
        self._train_anomaly_detector(historical_data)
    
    def _train_anomaly_detector(self, data: pd.DataFrame):
        """训练异常检测模型"""
        feature_cols = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic', 'steps', 'sleep_hours']
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return
        
        X = data[available_cols].dropna()
        if len(X) < 10:
            return
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_scaled)
    
    def check_threshold_anomaly(self, indicator: str, value: float) -> Tuple[bool, str, float]:
        """检查固定阈值异常"""
        is_anomaly = False
        severity = "low"
        confidence = 0.5
        
        if indicator == "heart_rate":
            low = self.config["heart_rate"]["low_threshold"]
            high = self.config["heart_rate"]["high_threshold"]
            if value < low:
                is_anomaly = True
                severity = "medium" if value > low - 10 else "high"
                confidence = min(abs(value - low) / 20, 1.0)
            elif value > high:
                is_anomaly = True
                severity = "medium" if value < high + 20 else "high"
                confidence = min(abs(value - high) / 30, 1.0)
        
        elif indicator == "blood_oxygen":
            low = self.config["blood_oxygen"]["low_threshold"]
            if value < low:
                is_anomaly = True
                severity = "high" if value < low - 5 else "medium"
                confidence = min(abs(value - low) / 10, 1.0)
        
        elif indicator == "systolic":
            low = self.config["blood_pressure"]["systolic_low"]
            high = self.config["blood_pressure"]["systolic_high"]
            if value < low:
                is_anomaly = True
                severity = "medium"
                confidence = min(abs(value - low) / 20, 1.0)
            elif value > high:
                is_anomaly = True
                severity = "high" if value > high + 20 else "medium"
                confidence = min(abs(value - high) / 30, 1.0)
        
        elif indicator == "diastolic":
            low = self.config["blood_pressure"]["diastolic_low"]
            high = self.config["blood_pressure"]["diastolic_high"]
            if value < low:
                is_anomaly = True
                severity = "medium"
                confidence = min(abs(value - low) / 10, 1.0)
            elif value > high:
                is_anomaly = True
                severity = "high" if value > high + 10 else "medium"
                confidence = min(abs(value - high) / 20, 1.0)
        
        elif indicator == "steps":
            low = self.config["activity"]["low_steps_threshold"]
            if value < low:
                is_anomaly = True
                severity = "low"
                confidence = min(abs(value - low) / 500, 1.0)
        
        elif indicator == "sleep_hours":
            min_h = self.config["sleep"]["min_hours"]
            max_h = self.config["sleep"]["max_hours"]
            if value < min_h:
                is_anomaly = True
                severity = "low"
                confidence = min(abs(value - min_h) / 2, 1.0)
            elif value > max_h:
                is_anomaly = True
                severity = "low"
                confidence = min(abs(value - max_h) / 3, 1.0)
        
        return is_anomaly, severity, confidence
    
    def check_baseline_anomaly(self, indicator: str, value: float) -> Tuple[bool, float]:
        """检查相对于个性化基线的异常"""
        if indicator not in self.baseline:
            return False, 0.0
        
        baseline_info = self.baseline[indicator]
        mean = baseline_info['mean']
        std = baseline_info['std']
        
        if std == 0:
            return False, 0.0
        
        # Z-score异常检测
        z_score = abs((value - mean) / std)
        
        if z_score > 3:
            return True, min(z_score / 5, 1.0)
        elif z_score > 2:
            return True, min(z_score / 4, 1.0)
        
        return False, 0.0
    
    def detect_anomaly(self, health_data: Dict) -> List[HealthAnomaly]:
        """检测健康数据异常"""
        anomalies = []
        timestamp = health_data.get('timestamp', datetime.now().isoformat())
        
        # 检查各指标
        indicators = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic', 'steps', 'sleep_hours']
        
        for indicator in indicators:
            if indicator not in health_data:
                continue
            
            value = health_data[indicator]
            if value is None or np.isnan(value):
                continue
            
            # 1. 固定阈值检测
            is_threshold_anomaly, severity, confidence = self.check_threshold_anomaly(indicator, value)
            
            if is_threshold_anomaly:
                # 获取阈值范围
                threshold = self._get_threshold_range(indicator)
                
                anomalies.append(HealthAnomaly(
                    timestamp=timestamp,
                    indicator=indicator,
                    value=value,
                    threshold=threshold,
                    severity=severity,
                    description=f"{self._get_indicator_name(indicator)}异常: {value} {self._get_unit(indicator)}",
                    confidence=confidence
                ))
            
            # 2. 个性化基线检测
            is_baseline_anomaly, baseline_conf = self.check_baseline_anomaly(indicator, value)
            
            if is_baseline_anomaly and not is_threshold_anomaly:
                # 相对于个人基线的异常
                anomalies.append(HealthAnomaly(
                    timestamp=timestamp,
                    indicator=indicator,
                    value=value,
                    threshold=(self.baseline[indicator]['q25'], self.baseline[indicator]['q75']),
                    severity="low",
                    description=f"{self._get_indicator_name(indicator)}偏离个人基线",
                    confidence=baseline_conf * 0.7
                ))
        
        # 3. 多指标联合异常检测
        if self.isolation_forest is not None:
            multivariate_anomaly = self._check_multivariate_anomaly(health_data)
            if multivariate_anomaly:
                anomalies.append(HealthAnomaly(
                    timestamp=timestamp,
                    indicator="multivariate",
                    value=0,
                    threshold=(0, 0),
                    severity="medium",
                    description="多指标综合异常",
                    confidence=0.6
                ))
        
        return anomalies
    
    def _check_multivariate_anomaly(self, health_data: Dict) -> bool:
        """多变量联合异常检测"""
        feature_cols = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic', 'steps', 'sleep_hours']
        available_cols = [col for col in feature_cols if col in health_data and health_data[col] is not None]
        
        if len(available_cols) < 2:
            return False
        
        X = np.array([[health_data.get(col, 0) for col in available_cols]])
        
        try:
            X_scaled = self.scaler.transform(X)
            prediction = self.isolation_forest.predict(X_scaled)
            return prediction[0] == -1
        except:
            return False
    
    def _get_threshold_range(self, indicator: str) -> Tuple[float, float]:
        """获取指标的阈值范围"""
        if indicator == "heart_rate":
            return (self.config["heart_rate"]["low_threshold"], 
                    self.config["heart_rate"]["high_threshold"])
        elif indicator == "blood_oxygen":
            return (self.config["blood_oxygen"]["low_threshold"], 100)
        elif indicator == "systolic":
            return (self.config["blood_pressure"]["systolic_low"], 
                    self.config["blood_pressure"]["systolic_high"])
        elif indicator == "diastolic":
            return (self.config["blood_pressure"]["diastolic_low"], 
                    self.config["blood_pressure"]["diastolic_high"])
        elif indicator == "steps":
            return (self.config["activity"]["low_steps_threshold"], 50000)
        elif indicator == "sleep_hours":
            return (self.config["sleep"]["min_hours"], self.config["sleep"]["max_hours"])
        return (0, 0)
    
    def _get_indicator_name(self, indicator: str) -> str:
        """获取指标中文名称"""
        names = {
            "heart_rate": "心率",
            "blood_oxygen": "血氧",
            "systolic": "收缩压",
            "diastolic": "舒张压",
            "steps": "步数",
            "sleep_hours": "睡眠时长",
        }
        return names.get(indicator, indicator)
    
    def _get_unit(self, indicator: str) -> str:
        """获取指标单位"""
        units = {
            "heart_rate": "bpm",
            "blood_oxygen": "%",
            "systolic": "mmHg",
            "diastolic": "mmHg",
            "steps": "步",
            "sleep_hours": "小时",
        }
        return units.get(indicator, "")
    
    def analyze_trend(self, historical_data: pd.DataFrame, 
                      indicator: str, days: int = 7) -> Dict:
        """分析健康指标趋势"""
        if indicator not in historical_data.columns:
            return {"error": f"未找到指标: {indicator}"}
        
        # 获取最近N天的数据
        recent_data = historical_data.tail(days)
        values = recent_data[indicator].dropna()
        
        if len(values) < 3:
            return {"error": "数据不足"}
        
        # 计算趋势
        mean_val = values.mean()
        std_val = values.std()
        trend = "stable"
        
        # 简单趋势判断
        if len(values) >= 3:
            first_half = values[:len(values)//2].mean()
            second_half = values[len(values)//2:].mean()
            change_ratio = (second_half - first_half) / (first_half + 1e-6)
            
            if change_ratio > 0.1:
                trend = "increasing"
            elif change_ratio < -0.1:
                trend = "decreasing"
        
        return {
            "indicator": indicator,
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(values.min()),
            "max": float(values.max()),
            "trend": trend,
            "days_analyzed": len(values),
        }
    
    def generate_health_summary(self, health_data: pd.DataFrame, 
                                days: int = 7) -> Dict:
        """生成健康摘要"""
        summary = {
            "period_days": days,
            "indicators": {},
            "anomalies_count": 0,
            "alerts": [],
        }
        
        indicators = ['heart_rate', 'blood_oxygen', 'systolic', 'diastolic', 'steps', 'sleep_hours']
        
        for indicator in indicators:
            if indicator in health_data.columns:
                trend = self.analyze_trend(health_data, indicator, days)
                if "error" not in trend:
                    summary["indicators"][indicator] = trend
        
        return summary


def generate_sample_health_data(days: int = 7) -> pd.DataFrame:
    """生成模拟健康数据"""
    np.random.seed(42)
    
    n_samples = days * 24  # 每小时一个样本
    
    data = {
        'timestamp': pd.date_range(end=datetime.now(), periods=n_samples, freq='H'),
        'heart_rate': np.random.normal(75, 10, n_samples).clip(50, 120),
        'blood_oxygen': np.random.normal(96, 2, n_samples).clip(88, 100),
        'systolic': np.random.normal(125, 15, n_samples).clip(90, 180),
        'diastolic': np.random.normal(80, 10, n_samples).clip(50, 120),
        'steps': np.random.exponential(200, n_samples).clip(0, 5000),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 12),
    }
    
    # 添加一些异常值
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data['heart_rate'][anomaly_indices[:len(anomaly_indices)//2]] = np.random.uniform(130, 150, len(anomaly_indices)//2)
    data['blood_oxygen'][anomaly_indices[len(anomaly_indices)//2:]] = np.random.uniform(80, 88, len(anomaly_indices) - len(anomaly_indices)//2)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 测试代码
    print("生理数据监测模块测试")
    monitor = HealthMonitor()
    print("模块初始化成功")
    
    # 生成测试数据
    test_data = generate_sample_health_data(7)
    print(f"生成测试数据: {len(test_data)} 条记录")
    
    # 设置基线
    monitor.set_baseline(test_data)
    print(f"建立个性化基线: {list(monitor.baseline.keys())}")
    
    # 检测异常
    test_record = {
        'timestamp': datetime.now().isoformat(),
        'heart_rate': 135,
        'blood_oxygen': 85,
        'systolic': 130,
        'diastolic': 85,
        'steps': 300,
        'sleep_hours': 6,
    }
    
    anomalies = monitor.detect_anomaly(test_record)
    print(f"检测到异常: {len(anomalies)} 个")
    for a in anomalies:
        print(f"  - {a.description}, 严重程度: {a.severity}")
