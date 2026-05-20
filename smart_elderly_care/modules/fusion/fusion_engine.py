"""
多模态融合决策模块
融合视频、音频、生理数据、用药信息进行综合风险评估
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskAssessment:
    """风险评估结果"""
    timestamp: str
    risk_level: RiskLevel
    risk_score: float
    contributing_factors: List[Dict]
    recommendation: str
    requires_immediate_action: bool


@dataclass
class ModalityResult:
    """单模态分析结果"""
    modality: str
    confidence: float
    risk_score: float
    events: List[Dict]
    timestamp: str


class FusionEngine:
    """多模态融合引擎"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            "weights": {
                "video": 0.4,
                "audio": 0.25,
                "health": 0.25,
                "medication": 0.1,
            },
            "risk_levels": {
                "low": {"min": 0, "max": 0.4},
                "medium": {"min": 0.4, "max": 0.7},
                "high": {"min": 0.7, "max": 1.0},
            }
        }
        
        # 历史评估记录
        self.assessment_history: List[RiskAssessment] = []
        
        # 事件缓存（用于时间窗口融合）
        self.event_buffer = {
            "video": [],
            "audio": [],
            "health": [],
            "medication": [],
        }
        
        # 时间窗口（秒）
        self.time_window = 60
    
    def add_modality_result(self, result: ModalityResult):
        """添加单模态分析结果"""
        self.event_buffer[result.modality].append({
            "confidence": result.confidence,
            "risk_score": result.risk_score,
            "events": result.events,
            "timestamp": result.timestamp,
        })
        
        # 清理过期事件
        self._clean_event_buffer()
    
    def _clean_event_buffer(self):
        """清理过期的事件"""
        now = datetime.now()
        for modality in self.event_buffer:
            self.event_buffer[modality] = [
                e for e in self.event_buffer[modality]
                if (now - datetime.fromisoformat(e["timestamp"])).total_seconds() < self.time_window
            ]
    
    def calculate_fused_risk(self) -> RiskAssessment:
        """计算融合后的风险评分"""
        weights = self.config["weights"]
        total_score = 0
        total_weight = 0
        contributing_factors = []
        
        # 计算各模态加权风险分数
        for modality, weight in weights.items():
            events = self.event_buffer.get(modality, [])
            
            if events:
                # 取最近事件的最高风险分数
                max_risk = max(e["risk_score"] for e in events)
                avg_confidence = np.mean([e["confidence"] for e in events])
                
                # 加权计算
                modality_score = max_risk * avg_confidence * weight
                total_score += modality_score
                total_weight += weight
                
                # 记录贡献因素
                if max_risk > 0.3:
                    contributing_factors.append({
                        "modality": modality,
                        "risk_score": max_risk,
                        "confidence": avg_confidence,
                        "weight": weight,
                        "events": [e["events"] for e in events if e["risk_score"] > 0.3],
                    })
        
        # 归一化风险分数
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0
        
        # 确定风险等级
        risk_level = self._determine_risk_level(final_score)
        
        # 生成建议
        recommendation = self._generate_recommendation(risk_level, contributing_factors)
        
        # 创建评估结果
        assessment = RiskAssessment(
            timestamp=datetime.now().isoformat(),
            risk_level=risk_level,
            risk_score=round(final_score, 3),
            contributing_factors=contributing_factors,
            recommendation=recommendation,
            requires_immediate_action=(risk_level == RiskLevel.HIGH),
        )
        
        # 保存历史
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """根据分数确定风险等级"""
        levels = self.config["risk_levels"]
        
        if score >= levels["high"]["min"]:
            return RiskLevel.HIGH
        elif score >= levels["medium"]["min"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendation(self, risk_level: RiskLevel, 
                                 factors: List[Dict]) -> str:
        """生成处理建议"""
        recommendations = {
            RiskLevel.LOW: "记录到系统，写入周报，持续观察。",
            RiskLevel.MEDIUM: "推送通知给子女，建议关注老人状态。",
            RiskLevel.HIGH: "立即报警！通知子女或照护人员，必要时联系急救。",
        }
        
        base_recommendation = recommendations[risk_level]
        
        # 根据具体因素补充建议
        additional = []
        for factor in factors:
            modality = factor["modality"]
            if modality == "video" and factor["risk_score"] > 0.7:
                additional.append("检测到疑似跌倒，建议立即确认老人状态。")
            elif modality == "audio" and factor["risk_score"] > 0.6:
                additional.append("检测到异常声音，建议确认是否有意外发生。")
            elif modality == "health" and factor["risk_score"] > 0.6:
                additional.append("生理指标异常，建议关注健康状况。")
            elif modality == "medication" and factor["risk_score"] > 0.5:
                additional.append("用药异常，建议确认服药情况。")
        
        if additional:
            return base_recommendation + " " + " ".join(additional)
        return base_recommendation
    
    def assess_fall_event(self, video_result: Optional[Dict] = None,
                         audio_result: Optional[Dict] = None,
                         health_result: Optional[Dict] = None) -> RiskAssessment:
        """
        专门针对跌倒事件的综合评估
        
        融合视频跌倒检测、音频撞击声、生理指标变化
        """
        # 重置事件缓存
        self.event_buffer = {"video": [], "audio": [], "health": [], "medication": []}
        
        # 添加各模态结果
        if video_result:
            self.add_modality_result(ModalityResult(
                modality="video",
                confidence=video_result.get("confidence", 0.5),
                risk_score=video_result.get("risk_score", 0.5),
                events=video_result.get("events", []),
                timestamp=datetime.now().isoformat(),
            ))
        
        if audio_result:
            self.add_modality_result(ModalityResult(
                modality="audio",
                confidence=audio_result.get("confidence", 0.5),
                risk_score=audio_result.get("risk_score", 0.5),
                events=audio_result.get("events", []),
                timestamp=datetime.now().isoformat(),
            ))
        
        if health_result:
            self.add_modality_result(ModalityResult(
                modality="health",
                confidence=health_result.get("confidence", 0.5),
                risk_score=health_result.get("risk_score", 0.5),
                events=health_result.get("events", []),
                timestamp=datetime.now().isoformat(),
            ))
        
        return self.calculate_fused_risk()
    
    def get_risk_trend(self, hours: int = 24) -> Dict:
        """获取风险趋势"""
        now = datetime.now()
        recent_assessments = [
            a for a in self.assessment_history
            if (now - datetime.fromisoformat(a.timestamp)).total_seconds() < hours * 3600
        ]
        
        if not recent_assessments:
            return {"trend": "no_data", "average_score": 0}
        
        scores = [a.risk_score for a in recent_assessments]
        
        # 计算趋势
        if len(scores) >= 3:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "average_score": round(np.mean(scores), 3),
            "max_score": round(max(scores), 3),
            "assessment_count": len(recent_assessments),
            "high_risk_count": sum(1 for a in recent_assessments if a.risk_level == RiskLevel.HIGH),
        }


class ActiveConfirmation:
    """主动确认机制"""
    
    def __init__(self):
        self.pending_confirmations = []
        self.confirmation_timeout = 60  # 秒
    
    def create_confirmation(self, event_type: str, message: str) -> Dict:
        """创建确认请求"""
        confirmation = {
            "id": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "created_at": datetime.now(),
            "status": "pending",
            "response": None,
        }
        self.pending_confirmations.append(confirmation)
        return confirmation
    
    def process_response(self, confirmation_id: str, response: str) -> Dict:
        """处理用户响应"""
        for conf in self.pending_confirmations:
            if conf["id"] == confirmation_id:
                conf["status"] = "responded"
                conf["response"] = response
                conf["responded_at"] = datetime.now()
                
                # 根据响应调整风险
                if response in ["不需要", "没事", "好的"]:
                    risk_adjustment = -0.3
                elif response in ["需要帮助", "救命", "疼"]:
                    risk_adjustment = 0.3
                else:
                    risk_adjustment = 0
                
                return {
                    "status": "confirmed",
                    "risk_adjustment": risk_adjustment,
                    "original_event": conf["event_type"],
                }
        
        return {"status": "not_found"}
    
    def check_timeout(self) -> List[Dict]:
        """检查超时未响应的确认"""
        now = datetime.now()
        timed_out = []
        
        for conf in self.pending_confirmations:
            if conf["status"] == "pending":
                elapsed = (now - conf["created_at"]).total_seconds()
                if elapsed > self.confirmation_timeout:
                    conf["status"] = "timeout"
                    timed_out.append(conf)
        
        return timed_out
    
    def get_voice_prompt(self, event_type: str) -> str:
        """生成语音提示"""
        prompts = {
            "fall": "检测到您可能摔倒了，请问您是否需要帮助？",
            "long_stillness": "检测到您长时间没有移动，请问您还好吗？",
            "health_anomaly": "检测到您的健康指标有异常，请问您感觉如何？",
            "help_call": "检测到呼救声，请问您需要帮助吗？",
        }
        return prompts.get(event_type, "检测到异常情况，请问您需要帮助吗？")


if __name__ == "__main__":
    # 测试代码
    print("多模态融合决策模块测试")
    engine = FusionEngine()
    print("模块初始化成功")
    
    # 模拟跌倒事件
    video_result = {
        "confidence": 0.85,
        "risk_score": 0.8,
        "events": [{"type": "fall", "description": "检测到疑似跌倒"}],
    }
    
    audio_result = {
        "confidence": 0.7,
        "risk_score": 0.6,
        "events": [{"type": "impact", "description": "检测到撞击声"}],
    }
    
    health_result = {
        "confidence": 0.6,
        "risk_score": 0.5,
        "events": [{"type": "heart_rate_high", "description": "心率异常升高"}],
    }
    
    assessment = engine.assess_fall_event(video_result, audio_result, health_result)
    
    print(f"\n风险评估结果:")
    print(f"  风险等级: {assessment.risk_level.value}")
    print(f"  风险分数: {assessment.risk_score}")
    print(f"  建议: {assessment.recommendation}")
    print(f"  需要立即行动: {assessment.requires_immediate_action}")
    
    # 测试主动确认
    print("\n主动确认机制测试:")
    confirmation = ActiveConfirmation()
    conf = confirmation.create_confirmation("fall", "检测到疑似跌倒")
    print(f"  创建确认: {conf['message']}")
    print(f"  语音提示: {confirmation.get_voice_prompt('fall')}")
