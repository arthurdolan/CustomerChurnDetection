"""
Mocked retention action handlers.
Simulates customer retention actions without actual external API calls.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict


class RetentionActionTracker:
    """Tracks retention actions taken for customers."""

    def __init__(self):
        self.actions_log = []

    def log_action(self, customer_id: str, action_type: str, status: str = "completed"):
        """Log a retention action."""
        self.actions_log.append(
            {
                "customer_id": customer_id,
                "action_type": action_type,
                "status": status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def get_customer_actions(self, customer_id: str) -> list:
        """Get all actions for a specific customer."""
        return [a for a in self.actions_log if a["customer_id"] == customer_id]

    def get_all_actions(self) -> pd.DataFrame:
        """Get all logged actions as a DataFrame."""
        if not self.actions_log:
            return pd.DataFrame(
                columns=["customer_id", "action_type", "status", "timestamp"]
            )
        return pd.DataFrame(self.actions_log)


# Global action tracker instance
action_tracker = RetentionActionTracker()


def send_retention_email(customer_id: str) -> Dict[str, str]:
    """Mock function to send a retention email."""
    action_tracker.log_action(customer_id, "email", "completed")
    return {
        "status": "success",
        "message": f"✅ Retention email sent to customer {customer_id}",
        "details": 'Email template: "We miss you! Here\'s a special offer just for you."',
    }


def offer_discount(customer_id: str, discount_percent: float = 15.0) -> Dict[str, str]:
    """Mock function to offer a discount."""
    action_tracker.log_action(customer_id, f"discount_{discount_percent}%", "completed")
    return {
        "status": "success",
        "message": f"✅ {discount_percent}% discount code generated for customer {customer_id}",
        "details": f"Discount code: SAVE{discount_percent}-{customer_id[:6].upper()}",
    }


def flag_for_call(customer_id: str) -> Dict[str, str]:
    """Mock function to flag customer for customer success call."""
    action_tracker.log_action(customer_id, "call", "pending")
    return {
        "status": "success",
        "message": f"✅ Customer {customer_id} flagged for customer success call",
        "details": "Added to priority queue. Expected callback within 24 hours.",
    }


def prioritize_success_call(customer_id: str) -> Dict[str, str]:
    """Mock function for priority success call."""
    action_tracker.log_action(customer_id, "priority_call", "pending")
    return {
        "status": "success",
        "message": f"✅ Priority call queued for customer {customer_id}",
        "details": "Assigned to senior retention specialist.",
    }


def send_personalized_retention_email(customer_id: str) -> Dict[str, str]:
    """Mock function for a personalized retention email."""
    action_tracker.log_action(customer_id, "personalized_email", "completed")
    return {
        "status": "success",
        "message": f"✅ Personalized retention email sent to customer {customer_id}",
        "details": "Personalized template with recent activity and tailored offer.",
    }


def flag_short_term_monitoring(customer_id: str) -> Dict[str, str]:
    """Mock function to flag account for monitoring."""
    action_tracker.log_action(customer_id, "short_term_monitoring", "active")
    return {
        "status": "success",
        "message": f"✅ Customer {customer_id} flagged for short-term monitoring",
        "details": "Monitoring period: 14 days.",
    }


def send_feature_education_email(customer_id: str) -> Dict[str, str]:
    """Mock function for feature education email."""
    action_tracker.log_action(customer_id, "feature_education_email", "completed")
    return {
        "status": "success",
        "message": f"✅ Feature education email sent to customer {customer_id}",
        "details": "Included quick-start guide and top features.",
    }


def send_usage_reminder(customer_id: str) -> Dict[str, str]:
    """Mock function for usage reminder."""
    action_tracker.log_action(customer_id, "usage_reminder", "completed")
    return {
        "status": "success",
        "message": f"✅ Usage reminder sent to customer {customer_id}",
        "details": "Inactivity nudge scheduled for next business day.",
    }


def add_to_experiment_cohort(customer_id: str) -> Dict[str, str]:
    """Mock function to add customer to retention experiment."""
    action_tracker.log_action(customer_id, "retention_experiment", "active")
    return {
        "status": "success",
        "message": f"✅ Customer {customer_id} added to retention experiment cohort",
        "details": "Cohort: Medium Risk Test Group A.",
    }


def issue_loyalty_reward(customer_id: str) -> Dict[str, str]:
    """Mock function to issue loyalty reward."""
    action_tracker.log_action(customer_id, "loyalty_reward", "completed")
    return {
        "status": "success",
        "message": f"✅ Loyalty reward issued to customer {customer_id}",
        "details": "Points credited to customer account.",
    }


def send_upsell_recommendation(customer_id: str) -> Dict[str, str]:
    """Mock function for upsell/cross-sell recommendation."""
    action_tracker.log_action(customer_id, "upsell_recommendation", "completed")
    return {
        "status": "success",
        "message": f"✅ Upsell recommendation sent to customer {customer_id}",
        "details": "Recommended complementary products.",
    }


def send_referral_incentive(customer_id: str) -> Dict[str, str]:
    """Mock function to send referral incentive."""
    action_tracker.log_action(customer_id, "referral_incentive", "completed")
    return {
        "status": "success",
        "message": f"✅ Referral incentive sent to customer {customer_id}",
        "details": "Referral code generated and delivered.",
    }


def request_feedback_review(customer_id: str) -> Dict[str, str]:
    """Mock function to request feedback/review."""
    action_tracker.log_action(customer_id, "feedback_request", "completed")
    return {
        "status": "success",
        "message": f"✅ Feedback request sent to customer {customer_id}",
        "details": "Survey link included.",
    }


def get_action_history(customer_id: Optional[str] = None) -> pd.DataFrame:
    """Get action history, optionally filtered by customer."""
    if customer_id:
        return pd.DataFrame(action_tracker.get_customer_actions(customer_id))
    return action_tracker.get_all_actions()
