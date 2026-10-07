import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app import mailer
from app.config import settings
from app.database import get_db
from app.models import EmailVerification, User
from app.schemas import UserCreate, UserLogin, Token, UserOut, UserUpdate, VerifyEmail, ResendCode
from app.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
log = logging.getLogger("heartbeat.auth")

CODE_TTL = timedelta(minutes=15)
CODE_MAX_ATTEMPTS = 5
RESEND_COOLDOWN = timedelta(seconds=60)

# The app tells this 403 apart from a wrong password by the detail.
NOT_VERIFIED = "EMAIL_NOT_VERIFIED"


def _hash_code(user_id: str, code: str) -> str:
    # Keyed and salted with the user, so a leaked table can't be reversed by
    # hashing all million codes once.
    return hmac.new(settings.SECRET_KEY.encode(), f"{user_id}:{code}".encode(), hashlib.sha256).hexdigest()


async def _send_code(db: AsyncSession, user: User) -> None:
    """Issue a fresh code (replacing any earlier one) and email it."""
    code = f"{secrets.randbelow(10**6):06d}"
    now = datetime.now()
    row = await db.get(EmailVerification, user.id)
    if row is None:
        row = EmailVerification(user_id=user.id)
        db.add(row)
    row.code_hash = _hash_code(user.id, code)
    row.expires_at = now + CODE_TTL
    row.attempts = 0
    row.sent_at = now
    await db.commit()
    try:
        await mailer.send_verification_code(user.email, user.full_name, code, int(CODE_TTL.total_seconds() // 60))
    except Exception:
        log.exception("could not send verification code to %s", user.email)
        raise HTTPException(status_code=503, detail="Không gửi được email xác thực, thử lại sau")


async def _user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


@router.post("/register", response_model=UserOut, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create the account and email it a code; it can't log in until
    /verify-email is called with that code."""
    user = await _user_by_email(db, data.email)
    if user and user.email_verified:
        raise HTTPException(status_code=400, detail="Email already registered")
    if user:
        # Signed up before but never verified: take the new details, since
        # nobody has proved they own the address yet.
        user.password_hash = hash_password(data.password)
        user.full_name = data.full_name
    else:
        user = User(email=data.email, password_hash=hash_password(data.password), full_name=data.full_name)
        db.add(user)
    await db.commit()
    await db.refresh(user)
    await _send_code(db, user)
    return user


@router.post("/verify-email", response_model=Token)
async def verify_email(data: VerifyEmail, db: AsyncSession = Depends(get_db)):
    """Check the emailed code; on success the account is verified and logged in."""
    invalid = HTTPException(status_code=400, detail="Mã xác thực không đúng")
    user = await _user_by_email(db, data.email)
    if user is None:
        raise invalid
    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email đã được xác thực, hãy đăng nhập")
    row = await db.get(EmailVerification, user.id)
    if row is None:
        raise invalid
    if row.expires_at < datetime.now() or row.attempts >= CODE_MAX_ATTEMPTS:
        raise HTTPException(status_code=400, detail="Mã đã hết hạn, hãy gửi lại mã mới")

    if not hmac.compare_digest(row.code_hash, _hash_code(user.id, data.code.strip())):
        row.attempts += 1
        await db.commit()
        raise invalid

    user.email_verified = True
    await db.delete(row)
    await db.commit()
    return Token(access_token=create_access_token(user.id))


@router.post("/resend-code", status_code=202)
async def resend_code(data: ResendCode, db: AsyncSession = Depends(get_db)):
    user = await _user_by_email(db, data.email)
    # Same answer for unknown and already-verified addresses.
    if user is None or user.email_verified:
        return {"ok": True}
    row = await db.get(EmailVerification, user.id)
    if row is not None and datetime.now() - row.sent_at < RESEND_COOLDOWN:
        wait = int((RESEND_COOLDOWN - (datetime.now() - row.sent_at)).total_seconds()) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Vui lòng đợi {wait} giây rồi gửi lại",
            headers={"Retry-After": str(wait)},
        )
    await _send_code(db, user)
    return {"ok": True}


@router.post("/login", response_model=Token)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await _user_by_email(db, data.email)
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Checked after the password, so this doesn't reveal which addresses
    # have unverified accounts.
    if not user.email_verified:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_VERIFIED)
    return Token(access_token=create_access_token(user.id))

@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user

@router.put("/me", response_model=UserOut)
async def update_me(data: UserUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
