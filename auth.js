/* =========================================================
   NairaLens — Auth
   Talks to the Flask backend's session-based auth endpoints.
   The browser holds a signed session cookie (credentials:
   'include'), so no token needs to be stored client-side.
   ========================================================= */
(function () {
  const API = window.NAIRALENS_API_BASE;

  async function request(path, options = {}) {
    const res = await fetch(API + path, {
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let data = {};
    try { data = await res.json(); } catch (e) { /* no body */ }
    if (!res.ok) {
      const err = new Error(data.error || `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function showToast(el, msg, kind) {
    if (!el) return;
    el.textContent = msg;
    el.classList.remove("ok", "err");
    el.classList.add("show", kind === "ok" ? "ok" : "err");
  }

  async function signUp({ name, email, password, confirm, watchlist }, toastEl) {
    if (!name || name.trim().length < 2) return showToast(toastEl, "Enter your full name.", "err");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email || "")) return showToast(toastEl, "Enter a valid email address.", "err");
    if (!password || password.length < 6) return showToast(toastEl, "Password must be at least 6 characters.", "err");
    if (password !== confirm) return showToast(toastEl, "Passwords do not match.", "err");

    try {
      await request("/auth/signup", {
        method: "POST",
        body: JSON.stringify({ name, email, password, watchlist }),
      });
      showToast(toastEl, "Account created. Redirecting to your dashboard…", "ok");
      setTimeout(() => { window.location.href = "dashboard.html"; }, 500);
    } catch (e) {
      showToast(toastEl, connectionAwareMessage(e), "err");
    }
  }

  async function signIn({ email, password }, toastEl) {
    try {
      await request("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      showToast(toastEl, "Signed in. Redirecting…", "ok");
      setTimeout(() => { window.location.href = "dashboard.html"; }, 400);
    } catch (e) {
      showToast(toastEl, connectionAwareMessage(e), "err");
    }
  }

  async function useDemoAccount(toastEl) {
    const email = "demo@nairalens.ai";
    const password = "demo1234";
    try {
      try {
        await request("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
      } catch (e) {
        if (e.status === 401) {
          await request("/auth/signup", {
            method: "POST",
            body: JSON.stringify({ name: "Demo Investor", email, password, watchlist: ["BTC", "ETH", "SOL"] }),
          });
        } else {
          throw e;
        }
      }
      if (toastEl) showToast(toastEl, "Loading demo workspace…", "ok");
      setTimeout(() => { window.location.href = "dashboard.html"; }, 350);
    } catch (e) {
      showToast(toastEl, connectionAwareMessage(e), "err");
    }
  }

  async function requireAuthOrRedirect() {
    try {
      const data = await request("/auth/me");
      return data.user;
    } catch (e) {
      window.location.href = "index.html";
      return null;
    }
  }

  async function signOut() {
    try { await request("/auth/logout", { method: "POST" }); } catch (e) { /* ignore */ }
    window.location.href = "index.html";
  }

  function connectionAwareMessage(e) {
    if (e.message && e.message.includes("Failed to fetch")) {
      return `Can't reach the backend at ${API}. Is the Flask server running?`;
    }
    return e.message || "Something went wrong.";
  }

  window.NairaAuth = { signUp, signIn, useDemoAccount, requireAuthOrRedirect, signOut, request, API_BASE: API };
})();
