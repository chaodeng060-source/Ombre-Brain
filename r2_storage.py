# ============================================================
# Module: R2 Object Storage Adapter (r2_storage.py)
# 模块：Cloudflare R2 对象存储适配层
#
# Uploads images (and other binary blobs) to Cloudflare R2
# using S3-compatible API, returns the public URL for retrieval.
# 通过 S3 兼容 API 把图片（及其他二进制 blob）上传到 Cloudflare R2，
# 返回可公开访问的 URL。
#
# Configuration via environment variables:
# 通过环境变量配置：
#   R2_ACCOUNT_ID         - Cloudflare account ID
#   R2_ACCESS_KEY_ID      - R2 API token access key
#   R2_SECRET_ACCESS_KEY  - R2 API token secret
#   R2_BUCKET_NAME        - bucket name, e.g. "ombre-brain-images"
#   R2_PUBLIC_URL         - public dev URL, e.g. "https://pub-xxx.r2.dev"
#
# If any required env var is missing, R2Storage operates in
# disabled mode — uploads silently no-op and return None,
# so the rest of Ombre Brain keeps working text-only.
# 任一必需环境变量缺失时，R2Storage 进入禁用模式，上传会安静返回 None，
# 其余记忆库功能不受影响。
# ============================================================

import os
import io
import base64
import logging
import mimetypes
from typing import Optional
from datetime import datetime

logger = logging.getLogger("ombre_brain.r2")

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO_AVAILABLE = True
except ImportError:
    BOTO_AVAILABLE = False
    logger.warning(
        "boto3 not installed — R2 image storage disabled. "
        "boto3 未安装 — R2 图片存储已禁用。"
    )


class R2Storage:
    """
    Cloudflare R2 storage adapter using S3-compatible API.

    Cloudflare R2 存储适配器，使用 S3 兼容 API。
    """

    def __init__(self):
        self.account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
        self.access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        self.secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket_name = os.environ.get("R2_BUCKET_NAME", "").strip()
        self.public_url = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")

        # Determine if R2 is fully configured / 判断 R2 是否完整配置
        self.enabled = bool(
            BOTO_AVAILABLE
            and self.account_id
            and self.access_key_id
            and self.secret_access_key
            and self.bucket_name
            and self.public_url
        )

        if not self.enabled:
            missing = []
            if not BOTO_AVAILABLE:
                missing.append("boto3 package")
            if not self.account_id:
                missing.append("R2_ACCOUNT_ID")
            if not self.access_key_id:
                missing.append("R2_ACCESS_KEY_ID")
            if not self.secret_access_key:
                missing.append("R2_SECRET_ACCESS_KEY")
            if not self.bucket_name:
                missing.append("R2_BUCKET_NAME")
            if not self.public_url:
                missing.append("R2_PUBLIC_URL")
            logger.info(
                f"R2 storage disabled, missing: {missing} / "
                f"R2 存储已禁用，缺失：{missing}"
            )
            self._client = None
            return

        # Build S3 client pointing at R2 endpoint / 构建指向 R2 的 S3 客户端
        endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        try:
            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name="auto",  # R2 ignores region, but boto3 requires one
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
            logger.info(
                f"R2 storage initialized: bucket={self.bucket_name} / "
                f"R2 存储已就绪：bucket={self.bucket_name}"
            )
        except Exception as e:
            logger.error(f"Failed to init R2 client / R2 客户端初始化失败: {e}")
            self._client = None
            self.enabled = False

    def upload_base64(
        self,
        b64_data: str,
        filename_hint: str = "image",
    ) -> Optional[str]:
        """
        Upload a base64-encoded blob (typically an image), return public URL.

        上传 base64 编码的二进制数据（通常是图片），返回公开访问 URL。

        Args:
            b64_data: base64 string (with or without data: URI prefix)
                      base64 字符串（含或不含 data: URI 前缀）
            filename_hint: human-readable filename hint, used for content-type
                          and the storage key. Will be sanitized + prefixed
                          with timestamp to avoid collisions.
                          可读文件名提示，用于推断 content-type 和生成存储键，
                          会被清洗并附加时间戳避免冲突。

        Returns:
            Public URL string on success, None on failure or when disabled.
            成功返回公开 URL 字符串，失败或禁用时返回 None。
        """
        if not self.enabled or not self._client:
            return None

        if not b64_data or not b64_data.strip():
            logger.warning("Empty base64 input / base64 输入为空")
            return None

        # --- Strip data URI prefix if present / 去除 data URI 前缀 ---
        # data:image/jpeg;base64,XXXXX  →  XXXXX
        if b64_data.startswith("data:"):
            try:
                b64_data = b64_data.split(",", 1)[1]
            except IndexError:
                logger.warning("Malformed data URI / data URI 格式错误")
                return None

        # --- Decode base64 / 解码 base64 ---
        try:
            blob = base64.b64decode(b64_data, validate=True)
        except Exception as e:
            logger.error(f"Base64 decode failed / base64 解码失败: {e}")
            return None

        if len(blob) == 0:
            logger.warning("Decoded blob is empty / 解码后数据为空")
            return None

        # --- Build storage key: timestamp_sanitized-name.ext ---
        # --- 构造存储键：时间戳_清洗后名称.扩展名 ---
        ext = self._guess_extension(filename_hint, blob)
        safe_name = self._sanitize_filename(filename_hint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = f"{timestamp}_{safe_name}{ext}"

        content_type = self._guess_content_type(ext)

        # --- Upload to R2 / 上传到 R2 ---
        try:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=blob,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"R2 upload failed / R2 上传失败: {e}")
            return None
        except Exception as e:
            logger.error(f"R2 upload unexpected error / R2 上传异常: {e}")
            return None

        public_url = f"{self.public_url}/{key}"
        logger.info(
            f"R2 uploaded {len(blob)} bytes → {public_url} / "
            f"R2 已上传 {len(blob)} 字节"
        )
        return public_url

    def delete(self, url: str) -> bool:
        """
        Delete an object by its public URL.
        Returns True on success or when object did not exist; False on error.

        通过公开 URL 删除对象。
        成功或对象本就不存在时返回 True，出错返回 False。
        """
        if not self.enabled or not self._client:
            return False
        if not url or not url.startswith(self.public_url):
            return False
        key = url[len(self.public_url):].lstrip("/")
        if not key:
            return False
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"R2 deleted: {key}")
            return True
        except Exception as e:
            logger.warning(f"R2 delete failed / R2 删除失败: {key}: {e}")
            return False

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Keep only safe chars, max 40 chars, no extension."""
        if not name:
            return "image"
        # strip extension if present / 去掉扩展名
        if "." in name:
            name = name.rsplit(".", 1)[0]
        # keep alphanumeric, dash, underscore / 只保留字母数字、连字符、下划线
        safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name)
        safe = safe.strip("-_")[:40]
        return safe or "image"

    @staticmethod
    def _guess_extension(filename_hint: str, blob: bytes) -> str:
        """
        Guess file extension from filename hint, fall back to magic bytes.
        从文件名提示推断扩展名，再降级到二进制魔数检测。
        """
        # Try filename / 先试文件名
        if filename_hint and "." in filename_hint:
            ext = "." + filename_hint.rsplit(".", 1)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"):
                return ext

        # Magic bytes / 魔数
        if blob.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if blob.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a"):
            return ".gif"
        if blob.startswith(b"RIFF") and len(blob) > 12 and blob[8:12] == b"WEBP":
            return ".webp"
        if blob.startswith(b"BM"):
            return ".bmp"

        return ".bin"

    @staticmethod
    def _guess_content_type(ext: str) -> str:
        ct, _ = mimetypes.guess_type(f"x{ext}")
        return ct or "application/octet-stream"


# Module-level singleton / 模块级单例
# Use this from server.py: from r2_storage import r2_storage
r2_storage = R2Storage()
