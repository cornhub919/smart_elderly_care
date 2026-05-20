"""
用药提醒与服药管理模块
管理用药计划、提醒服药、检测漏服
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import json


class MedicationStatus(Enum):
    """用药状态"""
    PENDING = "pending"  # 待服用
    TAKEN = "taken"  # 已服用
    MISSED = "missed"  # 漏服
    SKIPPED = "skipped"  # 跳过


@dataclass
class Medication:
    """药品信息"""
    name: str
    dosage: str
    frequency: int  # 每日次数
    times: List[str]  # 服药时间列表
    before_meal: bool = False
    notes: str = ""


@dataclass
class MedicationRecord:
    """服药记录"""
    medication_name: str
    scheduled_time: datetime
    actual_time: Optional[datetime]
    status: MedicationStatus
    dosage: str
    confirmed_by: str = "self"  # self, family, caregiver


class MedicationManager:
    """用药管理器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            "reminder_advance_minutes": 5,
            "missed_threshold_minutes": 30,
        }
        
        self.medications: List[Medication] = []
        self.records: List[MedicationRecord] = []
        self.today_schedule: List[Dict] = []
    
    def add_medication(self, name: str, dosage: str, frequency: int, 
                       times: List[str], before_meal: bool = False, 
                       notes: str = "") -> bool:
        """添加药品"""
        if len(times) != frequency:
            print(f"警告: 服药时间数量({len(times)})与频率({frequency})不匹配")
        
        medication = Medication(
            name=name,
            dosage=dosage,
            frequency=frequency,
            times=times,
            before_meal=before_meal,
            notes=notes
        )
        self.medications.append(medication)
        return True
    
    def remove_medication(self, name: str) -> bool:
        """移除药品"""
        for i, med in enumerate(self.medications):
            if med.name == name:
                self.medications.pop(i)
                return True
        return False
    
    def get_today_schedule(self) -> List[Dict]:
        """获取今日用药计划"""
        today = datetime.now().date()
        schedule = []
        
        for med in self.medications:
            for time_str in med.times:
                hour, minute = map(int, time_str.split(':'))
                scheduled_time = datetime.combine(today, datetime.min.time()) + \
                                timedelta(hours=hour, minutes=minute)
                
                # 检查是否已有记录
                record = self._find_record(med.name, scheduled_time)
                
                schedule.append({
                    "medication_name": med.name,
                    "dosage": med.dosage,
                    "scheduled_time": scheduled_time,
                    "status": record.status.value if record else "pending",
                    "before_meal": med.before_meal,
                    "notes": med.notes,
                })
        
        # 按时间排序
        schedule.sort(key=lambda x: x["scheduled_time"])
        self.today_schedule = schedule
        return schedule
    
    def _find_record(self, medication_name: str, 
                     scheduled_time: datetime) -> Optional[MedicationRecord]:
        """查找服药记录"""
        for record in self.records:
            if (record.medication_name == medication_name and 
                record.scheduled_time == scheduled_time):
                return record
        return None
    
    def confirm_medication(self, medication_name: str, 
                          scheduled_time: datetime,
                          confirmed_by: str = "self") -> bool:
        """确认服药"""
        record = self._find_record(medication_name, scheduled_time)
        
        if record:
            record.status = MedicationStatus.TAKEN
            record.actual_time = datetime.now()
            record.confirmed_by = confirmed_by
        else:
            # 创建新记录
            record = MedicationRecord(
                medication_name=medication_name,
                scheduled_time=scheduled_time,
                actual_time=datetime.now(),
                status=MedicationStatus.TAKEN,
                dosage=self._get_dosage(medication_name),
                confirmed_by=confirmed_by
            )
            self.records.append(record)
        
        return True
    
    def mark_missed(self, medication_name: str, 
                    scheduled_time: datetime) -> bool:
        """标记漏服"""
        record = MedicationRecord(
            medication_name=medication_name,
            scheduled_time=scheduled_time,
            actual_time=None,
            status=MedicationStatus.MISSED,
            dosage=self._get_dosage(medication_name),
        )
        self.records.append(record)
        return True
    
    def _get_dosage(self, medication_name: str) -> str:
        """获取药品剂量"""
        for med in self.medications:
            if med.name == medication_name:
                return med.dosage
        return ""
    
    def check_reminders(self) -> List[Dict]:
        """检查需要提醒的药品"""
        now = datetime.now()
        reminders = []
        
        schedule = self.get_today_schedule()
        
        for item in schedule:
            if item["status"] != "pending":
                continue
            
            scheduled_time = item["scheduled_time"]
            advance = timedelta(minutes=self.config["reminder_advance_minutes"])
            missed_threshold = timedelta(minutes=self.config["missed_threshold_minutes"])
            
            # 检查是否需要提醒
            if now >= scheduled_time - advance and now < scheduled_time + missed_threshold:
                reminders.append({
                    "type": "reminder",
                    "medication_name": item["medication_name"],
                    "dosage": item["dosage"],
                    "scheduled_time": scheduled_time,
                    "message": f"请按时服用{item['medication_name']}，剂量: {item['dosage']}",
                    "before_meal": item["before_meal"],
                })
            
            # 检查是否漏服
            elif now >= scheduled_time + missed_threshold:
                reminders.append({
                    "type": "missed",
                    "medication_name": item["medication_name"],
                    "dosage": item["dosage"],
                    "scheduled_time": scheduled_time,
                    "message": f"漏服警告: {item['medication_name']}，计划时间: {scheduled_time.strftime('%H:%M')}",
                })
                # 自动标记漏服
                self.mark_missed(item["medication_name"], scheduled_time)
        
        return reminders
    
    def check_duplicate_medication(self, medication_name: str, 
                                   hours: int = 2) -> bool:
        """检查是否重复服药"""
        now = datetime.now()
        threshold = now - timedelta(hours=hours)
        
        for record in self.records:
            if (record.medication_name == medication_name and 
                record.status == MedicationStatus.TAKEN and
                record.actual_time and 
                record.actual_time >= threshold):
                return True
        
        return False
    
    def get_adherence_stats(self, days: int = 7) -> Dict:
        """获取用药依从性统计"""
        now = datetime.now()
        start_date = now - timedelta(days=days)
        
        # 筛选时间范围内的记录
        relevant_records = [r for r in self.records 
                          if r.scheduled_time >= start_date]
        
        total_scheduled = len(relevant_records)
        taken = sum(1 for r in relevant_records if r.status == MedicationStatus.TAKEN)
        missed = sum(1 for r in relevant_records if r.status == MedicationStatus.MISSED)
        
        adherence_rate = (taken / total_scheduled * 100) if total_scheduled > 0 else 0
        
        # 按药品统计
        by_medication = {}
        for med in self.medications:
            med_records = [r for r in relevant_records 
                          if r.medication_name == med.name]
            med_taken = sum(1 for r in med_records if r.status == MedicationStatus.TAKEN)
            med_total = len(med_records)
            
            by_medication[med.name] = {
                "scheduled": med_total,
                "taken": med_taken,
                "missed": med_total - med_taken,
                "adherence_rate": (med_taken / med_total * 100) if med_total > 0 else 0,
            }
        
        return {
            "period_days": days,
            "total_scheduled": total_scheduled,
            "taken": taken,
            "missed": missed,
            "adherence_rate": round(adherence_rate, 1),
            "by_medication": by_medication,
        }
    
    def export_schedule(self) -> str:
        """导出用药计划为JSON"""
        data = {
            "medications": [
                {
                    "name": med.name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "times": med.times,
                    "before_meal": med.before_meal,
                    "notes": med.notes,
                }
                for med in self.medications
            ]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def import_schedule(self, json_str: str) -> bool:
        """从JSON导入用药计划"""
        try:
            data = json.loads(json_str)
            self.medications = []
            
            for med_data in data.get("medications", []):
                self.add_medication(
                    name=med_data["name"],
                    dosage=med_data["dosage"],
                    frequency=med_data["frequency"],
                    times=med_data["times"],
                    before_meal=med_data.get("before_meal", False),
                    notes=med_data.get("notes", ""),
                )
            
            return True
        except Exception as e:
            print(f"导入失败: {e}")
            return False


def create_sample_medication_plan() -> MedicationManager:
    """创建示例用药计划"""
    manager = MedicationManager()
    
    # 添加示例药品
    manager.add_medication(
        name="降压药",
        dosage="1片",
        frequency=2,
        times=["08:00", "20:00"],
        before_meal=False,
        notes="饭后服用，避免空腹"
    )
    
    manager.add_medication(
        name="降糖药",
        dosage="1片",
        frequency=1,
        times=["07:30"],
        before_meal=True,
        notes="饭前30分钟服用"
    )
    
    manager.add_medication(
        name="钙片",
        dosage="2片",
        frequency=1,
        times=["12:00"],
        before_meal=False,
        notes="午餐后服用"
    )
    
    return manager


if __name__ == "__main__":
    # 测试代码
    print("用药管理模块测试")
    manager = create_sample_medication_plan()
    
    print(f"\n已添加药品: {[m.name for m in manager.medications]}")
    
    # 获取今日计划
    schedule = manager.get_today_schedule()
    print(f"\n今日用药计划: {len(schedule)} 条")
    for item in schedule:
        print(f"  - {item['medication_name']}: {item['scheduled_time'].strftime('%H:%M')} ({item['status']})")
    
    # 获取依从性统计
    stats = manager.get_adherence_stats()
    print(f"\n用药依从性: {stats['adherence_rate']}%")
