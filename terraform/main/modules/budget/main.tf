# budget — a monthly cost budget with alert thresholds (a billing precondition).
#
# A budget does not cap spend; it emails the billing admins when actual cost crosses each
# threshold. This is the safety net the runbook requires before any cloud run. Scoped to the
# single project so it tracks only this deployment's cost.
#
# This module requires the quota-scoped google provider (billing_project set): the billingbudgets
# API rejects ADC requests with no quota project. The root passes it in as `google.billing_quota`.

terraform {
  required_providers {
    google = {
      source                = "hashicorp/google"
      configuration_aliases = [google.billing_quota]
    }
  }
}

variable "project_id" {
  type = string
}

variable "billing_account" {
  type = string
}

variable "amount_usd" {
  type = number
}

resource "google_billing_budget" "monthly" {
  provider        = google.billing_quota
  billing_account = var.billing_account
  display_name    = "scale-forecasting (${var.project_id})"

  budget_filter {
    projects               = ["projects/${var.project_id}"]
    calendar_period        = "MONTH"
    credit_types_treatment = "INCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = var.amount_usd
    }
  }

  # Alert at 50 / 90 / 100% of the budget. Default recipients are the billing-account admins
  # and users (no extra wiring needed for a solo/dev deployment).
  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }
}
