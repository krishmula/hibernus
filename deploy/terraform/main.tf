terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Credentials come from the standard AWS chain (env vars, ~/.aws profile, or instance role),
# the same chain boto3 and the Hibernus Go client use.
provider "aws" {
  region  = var.region
  profile = var.profile
}

# --- S3 bucket for restore points (what boto3's s3.create_bucket would do) ---
resource "aws_s3_bucket" "restore_points" {
  bucket = var.bucket_name

  # The existing bucket already carries this tag; declaring it stops Terraform removing it.
  tags = {
    Project = "hibernus"
  }
}

resource "aws_s3_bucket_public_access_block" "restore_points" {
  bucket                  = aws_s3_bucket.restore_points.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "restore_points" {
  bucket = aws_s3_bucket.restore_points.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Mirrors deploy/aws/lifecycle-test.json.
resource "aws_s3_bucket_lifecycle_configuration" "restore_points" {
  bucket = aws_s3_bucket.restore_points.id

  # Hibernus uploads large files in parts; clean up abandoned multipart uploads.
  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {
      prefix = ""
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  # Integration tests write under hibernus-test/; expire those objects after a day.
  rule {
    id     = "expire-integration-test-objects"
    status = "Enabled"
    filter {
      prefix = "hibernus-test/"
    }
    expiration {
      days = 1
    }
  }
}

# --- Least-privilege policy, mirrors deploy/aws/iam-policy.json ---
data "aws_iam_policy_document" "hibernus" {
  statement {
    sid = "HibernusRestorePointObjects"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
    ]
    resources = ["${aws_s3_bucket.restore_points.arn}/*"]
  }

  statement {
    sid       = "HibernusRestorePointBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.restore_points.arn]
  }
}

resource "aws_iam_policy" "hibernus" {
  name   = "hibernus-restore-points"
  policy = data.aws_iam_policy_document.hibernus.json
}

# Attach to an existing user so Hibernus no longer needs AdministratorAccess.
resource "aws_iam_user_policy_attachment" "hibernus" {
  count      = var.iam_user == "" ? 0 : 1
  user       = var.iam_user
  policy_arn = aws_iam_policy.hibernus.arn
}
