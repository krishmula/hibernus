variable "region" {
  type    = string
  default = "us-east-1"
}

variable "profile" {
  description = "AWS CLI profile; null uses the default credential chain"
  type        = string
  default     = null
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name for restore points"
  type        = string
  default     = "hibernus-restore-points-890b090de347a139bdec621af1"
}

variable "iam_user" {
  description = "Existing IAM user to attach the Hibernus policy to (empty = skip)"
  type        = string
  default     = ""
}
