from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    ForeignKey,
    Index,
    LargeBinary,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from . import config


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    # cascade delete: removing a person removes all their embeddings/thumbnails
    embeddings: Mapped[list["Embedding"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), index=True
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary)   # float32 bytes
    thumb_path: Mapped[str] = mapped_column(String(255), default="")

    person: Mapped["Person"] = relationship(back_populates="embeddings")


# Case-insensitive unique names, so concurrent enrolls can't duplicate a person.
Index("ix_people_name_ci", func.lower(Person.name), unique=True)


engine = create_engine(config.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
