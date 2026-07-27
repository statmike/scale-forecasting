# Bootstrap stage — pinned provider + Terraform versions.
#
# This stage uses LOCAL state (the state file lives next to these files on disk),
# because its whole job is to create the GCS bucket that the *main* stage will use
# for its remote state. That's the chicken-and-egg resolution (DESIGN §13.0): you
# can't store state in a bucket that doesn't exist yet, so the bucket is born here.

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}
