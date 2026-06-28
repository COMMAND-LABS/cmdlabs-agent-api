import uuid

from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.orm import relationship

from .database import Base
from .service_name import ServiceName


class Account(Base):
    __tablename__ = 'accounts'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    reset_token = Column(String)
    stripe_customer_id = Column(String, nullable=True)
    newsletter_subscribed = Column(Boolean, default=False, nullable=False)

    logins = relationship('Logins', back_populates='account')
    chat_sessions = relationship('ChatSession', back_populates='account')
    usage_credits = relationship('UsageCredits', back_populates='account')
    credentials = relationship('Credential', back_populates='account', cascade='all, delete-orphan')
    vector_db_logs = relationship('VectorDbIngestionLog', back_populates='account')
    api_keys = relationship('ApiKey', back_populates='account', cascade='all, delete-orphan')
    leads = relationship('Lead', back_populates='account', cascade='all, delete-orphan')
    prompts = relationship('Prompt', back_populates='account', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Account {self.email}>'

class Logins(Base):
    __tablename__ = 'logins'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'))
    created_at = Column(DateTime(timezone=True), default=func.now())
    ip_address = Column(String, nullable=False)
    similarity_score = Column(Double, default=False)

    account = relationship('Account', back_populates='logins')

    def __repr__(self):
        return f'<Login {self.login_time}>'

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(UUID, unique=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'), nullable=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    # Mirror of the ai-api column (migration owned by ai-api). The contact
    # binding is read here to scope the contact-chat agent's tools. If this
    # mirror drifts from ai-api, session.contact_id raises AttributeError only
    # on the contact-chat path — a test asserts this column exists.
    contact_id = Column(Integer, ForeignKey('contacts.id', ondelete='SET NULL'), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    title = Column(String)

    account = relationship('Account', back_populates='chat_sessions')
    agent = relationship('Agent', back_populates='chat_sessions')
    messages = relationship('ChatMessage', back_populates='session', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<ChatSession {self.session_id}>'

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey('chat_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    message = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=func.now())

    session = relationship('ChatSession', back_populates='messages')

    def __repr__(self):
        return f'<ChatMessage {self.id}>'


# ---------------------------------------------------------------------------
# CRM mirrors. The contacts / contact_events tables and their migrations are
# owned by kalygo3-ai-api; these are the agent-api mirror models so the
# contact-scoped agent tools can query the same shared database. Columns only
# (no relationships) to keep the mirror minimal. If these drift from ai-api,
# the contact-chat tools break — a test asserts the columns exist.
# ---------------------------------------------------------------------------

class Contact(Base):
    __tablename__ = 'contacts'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    first_name = Column(String(255), nullable=False)
    middle_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone = Column(String(50), nullable=True)
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f'<Contact {self.id}>'


class ContactEvent(Base):
    __tablename__ = 'contact_events'

    id = Column(Integer, primary_key=True, index=True)
    contact_id = Column(Integer, ForeignKey('contacts.id', ondelete='CASCADE'), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    occurred_at = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    def __repr__(self):
        return f'<ContactEvent {self.id}: {self.event_type} for contact {self.contact_id}>'


class UsageCredits(Base):
    __tablename__ = 'usage_credits'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    account = relationship('Account', back_populates='usage_credits')

    def __repr__(self):
        return f'<UsageCredits {self.account_id}: ${self.amount}>'

class Credential(Base):
    """
    Stores encrypted credentials for third-party services.

    The table supports multiple credential types:
    - API keys: Simple key-value (e.g., OpenAI API key)
    - Database connections: Host, port, username, password, database name
    - OAuth: Client ID, client secret, tokens
    - SSH keys: Private keys with optional passphrases
    - Certificates: Certificate data with optional private keys

    All credentials are stored in encrypted_data as encrypted JSON structures.
    """
    __tablename__ = 'credentials'
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    credential_type = Column(Enum(ServiceName, name='credential_type_enum'), nullable=False, index=True)
    auth_type = Column(String(50), nullable=False, index=True, default='api_key')
    credential_name = Column(String(255), nullable=True, index=True)

    # Encrypted storage (JSON structure, encrypted with Fernet)
    encrypted_data = Column(Text, nullable=False)

    # Non-sensitive metadata (e.g., display name, description, last_validated)
    credential_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    account = relationship('Account', back_populates='credentials')
    access_grants = relationship('CredentialAccessGrant', back_populates='credential', cascade='all, delete-orphan')

    def __repr__(self):
        name = self.credential_name or self.credential_type
        return f'<Credential {name} ({self.auth_type}) for account {self.account_id}>'


class CredentialAccessGrant(Base):
    """
    Shares a credential with EITHER an access group OR an individual account.

    Mirror of the ai-api definition (kept in parity so the synced
    credential_access.py service resolves identically in both services). Exactly
    one of access_group_id / grantee_account_id is set, enforced by the check
    constraint. Only the credential owner manages grants; recipients may USE the
    credential but never receive the plaintext.
    """
    __tablename__ = 'credential_access_grants'

    id = Column(Integer, primary_key=True, index=True)
    credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='CASCADE'), nullable=False, index=True)
    access_group_id = Column(Integer, ForeignKey('access_groups.id', ondelete='CASCADE'), nullable=True, index=True)
    grantee_account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            '(access_group_id IS NOT NULL)::int + (grantee_account_id IS NOT NULL)::int = 1',
            name='ck_credential_grant_exactly_one_target',
        ),
        Index(
            'uq_credential_grant_group',
            'credential_id', 'access_group_id',
            unique=True,
            postgresql_where=text('access_group_id IS NOT NULL'),
        ),
        Index(
            'uq_credential_grant_account',
            'credential_id', 'grantee_account_id',
            unique=True,
            postgresql_where=text('grantee_account_id IS NOT NULL'),
        ),
    )

    credential = relationship('Credential', back_populates='access_grants')
    access_group = relationship('AccessGroup')
    grantee = relationship('Account', foreign_keys=[grantee_account_id])

    def __repr__(self):
        target = f'group={self.access_group_id}' if self.access_group_id else f'account={self.grantee_account_id}'
        return f'<CredentialAccessGrant credential={self.credential_id} {target}>'


class CredentialDefault(Base):
    """
    A per-account, per-credential-type default selection (mirror of the ai-api
    definition). At most one default per credential_type per account. The
    credential_id FK cascades, so deleting a credential clears any default that
    pointed at it.
    """
    __tablename__ = 'credential_defaults'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    credential_type = Column(Enum(ServiceName, name='credential_type_enum', create_type=False), nullable=False, index=True)
    credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('account_id', 'credential_type', name='uq_credential_default_account_type'),
    )

    account = relationship('Account', foreign_keys=[account_id])
    credential = relationship('Credential', foreign_keys=[credential_id])

    def __repr__(self):
        return f'<CredentialDefault account={self.account_id} type={self.credential_type} -> credential={self.credential_id}>'


class VectorStore(Base):
    """
    A knowledge base (Pinecone index) owned by an account with explicit
    credential bindings (mirror of the ai-api definition — kept in parity so the
    synced vector_store_credentials.py resolves identically in both services).

    Nullable FKs fall back to the owner's default credential for that type when
    unset; see services/vector_store_credentials.py.
    """
    __tablename__ = 'vector_stores'

    id = Column(Integer, primary_key=True, index=True)
    owner_account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    index_name = Column(String, nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    pinecone_credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='SET NULL'), nullable=True, index=True)
    gcs_credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='SET NULL'), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('owner_account_id', 'index_name', name='uq_vector_store_owner_index'),
    )

    owner = relationship('Account', foreign_keys=[owner_account_id])
    pinecone_credential = relationship('Credential', foreign_keys=[pinecone_credential_id])
    gcs_credential = relationship('Credential', foreign_keys=[gcs_credential_id])

    def __repr__(self):
        return f'<VectorStore owner={self.owner_account_id} index={self.index_name}>'


class AccessGrant(Base):
    """
    Unified access grant (mirror of the ai-api definition — kept in parity so the
    synced access.py resolver behaves identically). A PRINCIPAL ('account'|'group')
    is granted a ROLE ('read'|'write'|'use') on a RESOURCE
    ('agent'|'vector_store'|'credential'). See ai-api models for full docs.
    """
    __tablename__ = 'access_grants'

    id = Column(Integer, primary_key=True, index=True)
    principal_type = Column(String(20), nullable=False)
    principal_id = Column(Integer, nullable=False)
    resource_type = Column(String(20), nullable=False)
    resource_id = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False, server_default='read')
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('principal_type', 'principal_id', 'resource_type', 'resource_id',
                         name='uq_access_grant_principal_resource'),
        CheckConstraint("principal_type IN ('account','group')", name='ck_access_grant_principal_type'),
        CheckConstraint("resource_type IN ('agent','vector_store','credential')", name='ck_access_grant_resource_type'),
        CheckConstraint("role IN ('read','write','use')", name='ck_access_grant_role'),
        Index('ix_access_grants_resource', 'resource_type', 'resource_id'),
        Index('ix_access_grants_principal', 'principal_type', 'principal_id'),
    )

    def __repr__(self):
        return (f'<AccessGrant {self.principal_type}={self.principal_id} '
                f'{self.role} {self.resource_type}={self.resource_id}>')


class ApiKeyStatus(str, Enum):
    """Enumeration of API key statuses."""
    ACTIVE = "active"
    REVOKED = "revoked"


class ApiKey(Base):
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)

    # Key storage: hash the full key, store prefix for display/lookup
    key_hash = Column(String, nullable=False, unique=True, index=True)
    key_prefix = Column(String, nullable=False, index=True)  # First 20 chars for display/lookup

    # Optional metadata
    name = Column(String, nullable=True)  # User-friendly name (e.g., "Website Chatbot")
    status = Column(PG_ENUM('active', 'revoked', name='api_key_status_enum', create_type=False), nullable=False, default=ApiKeyStatus.ACTIVE, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    account = relationship('Account', back_populates='api_keys')

    def __repr__(self):
        return f'<ApiKey {self.key_prefix}... for account {self.account_id}>'


class OperationType(str, Enum):
    """Enumeration of vector database operation types."""
    INGEST = "INGEST"
    DELETE = "DELETE"
    UPDATE = "UPDATE"


class OperationStatus(str, Enum):
    """Enumeration of vector database operation statuses."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"


class VectorDbIngestionLog(Base):
    __tablename__ = 'vector_db_ingestion_log'

    # Primary Key (UUID)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)

    # Operation Details
    # Note: Enum types are created in migration, using create_type=False here
    operation_type = Column(
        PG_ENUM('INGEST', 'DELETE', 'UPDATE', name='operation_type_enum', create_type=False),
        nullable=False,
        index=True
    )
    status = Column(
        PG_ENUM('SUCCESS', 'FAILED', 'PARTIAL', 'PENDING', name='operation_status_enum', create_type=False),
        nullable=False,
        index=True
    )

    # User/Account
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)

    # Vector Database Info
    provider = Column(String, nullable=False)  # 'pinecone', 'chroma', etc.
    index_name = Column(String, nullable=False, index=True)
    namespace = Column(String, nullable=True, index=True)

    # File Information
    filenames = Column(JSON, nullable=True)  # Array of filenames
    comment = Column(Text, nullable=True)

    # Vector Counts
    vectors_added = Column(Integer, default=0)
    vectors_deleted = Column(Integer, default=0)
    vectors_failed = Column(Integer, default=0)

    # Error Handling
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)

    # Batch Grouping
    batch_number = Column(String, nullable=True, index=True)  # UUID for grouping related operations

    # Relationships
    account = relationship('Account', back_populates='vector_db_logs')

    def __repr__(self):
        return f'<VectorDbIngestionLog {self.id} - {self.operation_type.value} - {self.status.value}>'


class Agent(Base):
    __tablename__ = 'agents'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    config = Column(JSON, nullable=True)  # JSONB in PostgreSQL, JSON in SQLAlchemy

    chat_sessions = relationship('ChatSession', back_populates='agent', cascade='all, delete-orphan')
    access_grants = relationship('AgentAccessGrant', back_populates='agent', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Agent {self.id}: {self.name}>'


# ---------------------------------------------------------------------------
# Access groups – allow agents to be shared with groups of accounts
# Tables are created/migrated from kalygo3-ai-api; these models are the
# agent-api mirror so SQLAlchemy can query the same shared database.
# ---------------------------------------------------------------------------

class AccessGroup(Base):
    """
    Named access group owned by an account.
    The owner can add/remove members and agents can be granted to the group.
    """
    __tablename__ = 'access_groups'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    owner_account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # NOTE: these ORM relationships are intentionally unused at runtime — all
    # access code uses explicit db.query(...).join(...).filter(...). They exist
    # only to document the schema, and their names may differ from ai-api's copy.
    owner = relationship('Account', foreign_keys=[owner_account_id])
    members = relationship('AccessGroupMember', back_populates='access_group', cascade='all, delete-orphan')
    agent_grants = relationship('AgentAccessGrant', back_populates='access_group', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<AccessGroup {self.id}: {self.name}>'


class AccessGroupMember(Base):
    """
    Membership link: an account belongs to an access group.
    Only the group owner can add/remove members (enforced at API layer).
    """
    __tablename__ = 'access_group_members'

    id = Column(Integer, primary_key=True, index=True)
    access_group_id = Column(Integer, ForeignKey('access_groups.id', ondelete='CASCADE'), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    # Mirror of the ai-api column ('admin' | 'member'). agent-api never reads it — the
    # access check only needs membership existence — but the model matches the live table.
    role = Column(String(50), nullable=False, server_default='member')
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    access_group = relationship('AccessGroup', back_populates='members')
    account = relationship('Account')

    __table_args__ = (
        UniqueConstraint('access_group_id', 'account_id', name='uq_access_group_members_group_account'),
    )

    def __repr__(self):
        return f'<AccessGroupMember group={self.access_group_id} account={self.account_id}>'


class AgentAccessGrant(Base):
    """
    Grant link: an agent is shared with an access group.
    Only the agent owner can create/revoke grants (enforced at API layer).
    """
    __tablename__ = 'agent_access_grants'

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True)
    access_group_id = Column(Integer, ForeignKey('access_groups.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    agent = relationship('Agent', back_populates='access_grants')
    access_group = relationship('AccessGroup', back_populates='agent_grants')

    __table_args__ = (
        UniqueConstraint('agent_id', 'access_group_id', name='uq_agent_access_grants_agent_group'),
    )

    def __repr__(self):
        return f'<AgentAccessGrant agent={self.agent_id} group={self.access_group_id}>'


class Lead(Base):
    """
    Stores lead/inquiry information.

    Leads are potential customers or inquiries captured through
    various channels (website forms, chat, etc.).
    """
    __tablename__ = 'leads'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    chat_session_id = Column(UUID, nullable=True, index=True)  # UUID of the chat session where lead was captured
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    account = relationship('Account', back_populates='leads')

    def __repr__(self):
        return f'<Lead {self.id}: {self.name}>'


class Prompt(Base):
    """
    Stores reusable prompt templates.

    Prompts are text templates that can be saved and reused
    across different agents or contexts.
    """
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    account = relationship('Account', back_populates='prompts')

    def __repr__(self):
        return f'<Prompt {self.id}: {self.name}>'


class EmailTemplate(Base):
    """
    Read-only mirror of the email_templates table managed by kalygo3-ai-api.
    The agent-api queries this table to fetch templates for rendering
    before queuing a sendHtmlEmailWithSes approval.
    """
    __tablename__ = 'email_templates'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    subject_template = Column(String(998), nullable=False)
    html_template = Column(Text, nullable=False)
    variables = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(),
                        onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f'<EmailTemplate {self.id}: {self.name}>'


class PendingToolApproval(Base):
    """
    Written by HITL-gated tools (e.g. send_email) when they want to queue an
    action for human review.  The approval REST endpoints in kalygo3-ai-api
    own the full lifecycle; the agent-api only ever *inserts* rows here.
    """
    __tablename__ = 'pending_tool_approvals'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('accounts.id', ondelete='CASCADE'), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='SET NULL'), nullable=True)
    chat_session_id = Column(Integer, ForeignKey('chat_sessions.id', ondelete='SET NULL'), nullable=True)

    tool_type = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default='pending')
    payload = Column(JSON, nullable=False)

    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f'<PendingToolApproval {self.id}: {self.tool_type} [{self.status}]>'
