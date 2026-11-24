output "bucket" {
  value = aws_s3_bucket.restore_points.id
}

output "remote_flag" {
  description = "Value for hib --remote"
  value       = "s3://${aws_s3_bucket.restore_points.id}"
}

output "policy_arn" {
  value = aws_iam_policy.hibernus.arn
}
