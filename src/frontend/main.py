# diagnostic test
import requests
import streamlit as st

st.set_page_config(page_title="Support Agent", page_icon="🤖")

BACKEND = "http://localhost:8000"


def api_post(endpoint, data=None, files=None, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = 300 if endpoint.startswith("/upload") or endpoint.startswith("/chat") else 30
    try:
        if files:
            r = requests.post(f"{BACKEND}{endpoint}", data=data, files=files, headers=headers, timeout=timeout)
        else:
            r = requests.post(f"{BACKEND}{endpoint}", json=data, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def init_state():
    defaults = {
        "page": "landing",
        "step": "choose_role",
        "role": None,
        "admin_id": None,
        "admin_code": None,
        "identity_table": None,
        "pk_column": None,
        "email_column": None,
        "access_token": None,
        "thread_id": None,
        "messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def go(page):
    st.session_state.page = page
    st.rerun()


def set_step(step):
    st.session_state.step = step
    st.rerun()


# ── Step 1: Choose role ─────────────────────────────────

def choose_role():
    st.title("🤖 Support Agent")
    st.write("Are you an admin or a customer?")
    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 👨‍💼 Admin")
        st.write("Manage your knowledge base, database, and view conversations.")
        if st.button("Admin", use_container_width=True, type="primary"):
            st.session_state.role = "admin"
            set_step("choose_action")
    with col2:
        st.markdown("### 👤 Customer")
        st.write("Ask questions about your account, orders, or company policies.")
        if st.button("Customer", use_container_width=True, type="primary"):
            st.session_state.role = "customer"
            set_step("choose_action")


# ── Step 2: Choose action ────────────────────────────────

def choose_action():
    role = st.session_state.role
    icon = "👨‍💼" if role == "admin" else "👤"
    label = "Admin" if role == "admin" else "Customer"

    st.title(f"{icon} {label}")
    st.write("Do you want to sign up or sign in?")
    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button("📝 Sign Up", use_container_width=True):
            set_step("signup")
    with col2:
        if st.button("🔑 Sign In", use_container_width=True):
            set_step("signin")

    st.write("")
    if st.button("← Back"):
        set_step("choose_role")


# ── Step 3: Forms ────────────────────────────────────────

def signup_form():
    role = st.session_state.role
    icon = "👨‍💼" if role == "admin" else "👤"
    label = "Admin" if role == "admin" else "Customer"

    st.title(f"{icon} {label} Sign Up")
    with st.form("signup_form"):
        email = st.text_input("Email", placeholder=f"{label.lower()}@company.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")

        extra = None
        if role == "customer":
            extra = st.text_input("Admin Code", placeholder="Enter admin code").upper()

        if st.form_submit_button("Create Account", use_container_width=True):
            if not email or not password:
                st.error("Please fill in all fields")
            elif password != confirm:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                if role == "admin":
                    r = api_post("/auth/admin/signup", {"email": email, "password": password})
                else:
                    r = api_post("/auth/user/signup", {
                        "email": email, "password": password, "admin_code": extra
                    })
                if "error" in r:
                    st.error(r["error"])
                else:
                    if role == "admin":
                        st.session_state.admin_code = r["admin_code"]
                        st.success(f"Admin Code: {r['admin_code']}")
                    else:
                        st.success("Account created!")
                    set_step("signin")

    st.write("")
    if st.button("← Back"):
        set_step("choose_action")


def signin_form():
    role = st.session_state.role
    icon = "👨‍💼" if role == "admin" else "👤"
    label = "Admin" if role == "admin" else "Customer"

    st.title(f"{icon} {label} Sign In")
    with st.form("signin_form"):
        email = st.text_input("Email", placeholder=f"{label.lower()}@company.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.form_submit_button("Sign In", use_container_width=True):
            if not email or not password:
                st.error("Please fill in all fields")
            else:
                if role == "admin":
                    r = api_post("/auth/admin/signin", {"email": email, "password": password})
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        st.session_state.admin_id = r["admin_id"]
                        st.session_state.admin_code = r["admin_code"]
                        st.session_state.database_url = r.get("database_url")
                        st.session_state.identity_table = r.get("identity_table")
                        st.session_state.pk_column = r.get("pk_column")
                        st.session_state.email_column = r.get("email_column")
                        go("admin_dashboard")
                else:
                    r = api_post("/auth/user/signin", {"email": email, "password": password})
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        st.session_state.admin_id = r["admin_id"]
                        st.session_state.access_token = r["access_token"]
                        go("customer_chat")

    st.write("")
    if st.button("← Back"):
        set_step("choose_action")


# ── Admin Dashboard ────────────────────────────────────────────────

def admin_dashboard():
    st.title("👨‍💼 Admin Dashboard")

    # Header with admin code display and logout
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Admin Code:** `{st.session_state.admin_code}`")
        st.caption("Share this code with customers to let them sign up")
    with col2:
        st.write(f"Admin ID: `{st.session_state.admin_id}`")
    with col3:
        if st.button("Logout", use_container_width=True):
            for k in ["admin_id", "user_id", "admin_code", "identity_table", "pk_column", "email_column", "thread_id", "messages", "role"]:
                st.session_state[k] = None if k != "messages" else []
            st.session_state.page = "landing"
            st.rerun()

    st.divider()

    # Database upload
    st.subheader("📊 Database Connection")
    db_url = st.session_state.get("database_url")
    if db_url:
        st.success(f"✅ Database already uploaded: `{db_url}`")
        st.write("Upload a new one to replace it:")
    else:
        st.write("No database uploaded yet.")

    with st.form("db"):
        url = st.text_input("Database URL",
                            value=db_url or "",
                            placeholder="postgresql://user:pass@host:5432/dbname")
        identity_table = st.text_input("Customer Table",
                                       value=st.session_state.get("identity_table", "customers"),
                                       placeholder="customers")
        pk_column = st.text_input("Primary Key Column",
                                  value=st.session_state.get("pk_column", "customer_id"),
                                  placeholder="customer_id")
        email_column = st.text_input("Email Column",
                                     value=st.session_state.get("email_column", "email"),
                                     placeholder="email")
        if st.form_submit_button("Save & Index Schema", use_container_width=True):
            if not url:
                st.error("Please enter a database URL")
            else:
                with st.spinner("Saving and indexing..."):
                    r = api_post("/upload/database", {
                        "admin_id": st.session_state.admin_id,
                        "database_url": url,
                        "identity_table": identity_table,
                        "pk_column": pk_column,
                        "email_column": email_column,
                    })
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        st.session_state.database_url = url
                        st.session_state.identity_table = identity_table
                        st.session_state.pk_column = pk_column
                        st.session_state.email_column = email_column
                        st.success(f"Indexed {len(r['table_ids'])} tables")

    # Knowledge base upload
    st.subheader("📚 Knowledge Base")
    if st.session_state.get("kb_uploaded"):
        st.success("✅ Knowledge base documents already uploaded.")
        st.write("Upload more to add to it:")
    else:
        st.write("No documents uploaded yet.")

    files = st.file_uploader("Upload documents (PDF, DOCX, TXT, etc.)",
                             accept_multiple_files=True,
                             type=["pdf", "docx", "txt", "md", "csv", "xlsx"])
    if st.button("Upload & Index", use_container_width=True, disabled=not files):
        with st.spinner(f"Processing {len(files)} file(s)..."):
            file_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            r = api_post("/upload/knowledge_base", {"admin_id": st.session_state.admin_id}, files=file_data)
            if "error" in r:
                st.error(r["error"])
            else:
                st.session_state.kb_uploaded = True
                st.success(f"Created {len(r['chunk_ids'])} chunks")


# ── Customer Chat ───────────────────────────────────────────────

def customer_chat():
    st.title("💬 Support Chat")

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("Ask me anything about your account, orders, or company policies.")
    with col2:
        if st.button("New Chat", use_container_width=True):
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("Logout", use_container_width=True):
            for k in ["admin_id", "user_id", "admin_code", "identity_table", "pk_column", "email_column", "thread_id", "messages", "role"]:
                st.session_state[k] = None if k != "messages" else []
            st.session_state.page = "landing"
            st.rerun()

    st.divider()

    # Chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            r = api_post("/chat/chat", {"message": prompt, "thread_id": st.session_state.thread_id},
                         token=st.session_state.access_token)
        if "error" in r:
            st.error(r["error"])
        else:
            st.session_state.thread_id = r["thread_id"]
            st.session_state.messages.append({"role": "assistant", "content": r["reply"]})
            st.rerun()


# ── Router ───────────────────────────────────────────────

if st.session_state.page == "landing":
    step = st.session_state.get("step", "choose_role")
    if step == "choose_role":
        choose_role()
    elif step == "choose_action":
        choose_action()
    elif step == "signup":
        signup_form()
    elif step == "signin":
        signin_form()
elif st.session_state.page == "admin_dashboard":
    admin_dashboard()
elif st.session_state.page == "customer_chat":
    customer_chat()