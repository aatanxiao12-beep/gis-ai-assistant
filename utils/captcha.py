"""
极简图形验证码（基于 fast-captcha）

- generate_captcha() → (image_base64, text)
- CaptchaStore: 内存存储，带过期时间，一次性验证
"""

import time

from fast_captcha import img_captcha


# ── 验证码生成 ──────────────────────────────────────────────

def generate_captcha(width: int = 140, height: int = 50) -> tuple[str, str]:
    """生成极简图形验证码

    Returns:
        (image_base64_png, answer_text)
    """
    b64_bytes, text = img_captcha(
        code_num=4,
        width=width,
        height=height,
        draw_lines=True,
        lines_num=3,
        draw_points=True,
        points_density=4,
        img_type="png",
        img_byte="base64",
    )
    return b64_bytes.decode() if isinstance(b64_bytes, bytes) else b64_bytes, text


# ── 验证码存储（内存 + TTL） ─────────────────────────────────

class CaptchaStore:
    """内存验证码存储，自动过期清理"""

    def __init__(self, ttl: int = 300):
        """
        Args:
            ttl: 验证码有效期（秒），默认 5 分钟
        """
        self._store: dict[str, tuple[str, float]] = {}  # id → (text, expire_at)
        self._ttl = ttl

    def put(self, captcha_id: str, text: str):
        """保存验证码"""
        self._store[captcha_id] = (text.upper(), time.time() + self._ttl)

    def verify(self, captcha_id: str, user_input: str) -> bool:
        """验证用户输入，验证成功后立即删除（一次性）"""
        entry = self._store.pop(captcha_id, None)
        if entry is None:
            return False
        text, expires = entry
        if time.time() > expires:
            return False
        return text == user_input.strip().upper()

    def cleanup(self):
        """清理过期条目（可定期调用）"""
        now = time.time()
        expired = [k for k, (_, t) in self._store.items() if now > t]
        for k in expired:
            self._store.pop(k, None)


# 全局单例
captcha_store = CaptchaStore(ttl=300)
