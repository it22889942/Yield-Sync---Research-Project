import { useEffect, useMemo, useState } from "react";

type Tab = "labour" | "equipment" | "pending";

type LabourAd = {
  id: string;
  name: string;
  location: string;
  labour_type: string;
  skill_level: string;
  hourly_rate: number;
  rating: number;
  jobs_completed: number;
  experience_years: number;
  available_day: string;
  available_time: string;
  season?: string;
  crop_type?: string;
  price_outlier?: boolean;
  price_median?: number | null;
  price_low_threshold?: number | null;
  price_high_threshold?: number | null;
  price_outlier_reason?: string;
};

type EquipmentAd = {
  id: string;
  equipment_type: string;
  for_crop: string;
  location: string;
  hourly_rate: number;
  daily_rate: number;
  rating: number;
  past_bookings: number;
  owner_name: string;
  available_day: string;
  available_time: string;
  condition: string;
  nearest_major_district?: string;
  hourly_price_outlier?: boolean;
  hourly_price_median?: number | null;
  hourly_price_low_threshold?: number | null;
  hourly_price_high_threshold?: number | null;
  hourly_price_outlier_reason?: string;
  daily_price_outlier?: boolean;
  daily_price_median?: number | null;
  daily_price_low_threshold?: number | null;
  daily_price_high_threshold?: number | null;
  daily_price_outlier_reason?: string;
};

/** Dev: use Vite proxy `/api` → backend. Prod: set VITE_API_BASE_URL or default localhost. */
function getApiBaseUrl(): string {
  const fromEnv = import.meta.env.VITE_API_BASE_URL;
  if (fromEnv && String(fromEnv).trim()) {
    return String(fromEnv).replace(/\/+$/, "");
  }
  if (import.meta.env.DEV) {
    return "/api";
  }
  return "http://localhost:5003/api";
}

const API_BASE_URL = getApiBaseUrl();

const FETCH_TIMEOUT_MS = 90_000;

async function fetchJsonWithTimeout<T>(
  url: string,
  init: RequestInit
): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      const text = await response.text();
      let detail = "";
      try {
        const j = JSON.parse(text) as { error?: string };
        detail = j.error ? ` — ${j.error}` : "";
      } catch {
        if (text) detail = ` — ${text.slice(0, 200)}`;
      }
      throw new Error(`HTTP ${response.status}${detail}`);
    }
    return (await response.json()) as T;
  } catch (e) {
    if (e instanceof Error) {
      if (e.name === "AbortError") {
        throw new Error(
          "Request timed out. The backend may be stuck on Firestore (check serviceAccountKey.json / network), or the server is not on port 5003."
        );
      }
      if (e.message.includes("Failed to fetch") || e.name === "TypeError") {
        throw new Error(
          "Cannot reach API. Start the Flask backend (python app.py) and use npm run dev so /api is proxied, or set VITE_API_BASE_URL."
        );
      }
    }
    throw e;
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchLabourAds(): Promise<LabourAd[]> {
  const data = await fetchJsonWithTimeout<{ items?: LabourAd[] }>(
    `${API_BASE_URL}/labour/search`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "", top_k: 500 }),
    }
  );
  return data.items ?? [];
}

async function fetchEquipmentAds(): Promise<EquipmentAd[]> {
  const data = await fetchJsonWithTimeout<{ items?: EquipmentAd[] }>(
    `${API_BASE_URL}/equipment/search`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "", top_k: 500 }),
    }
  );
  return data.items ?? [];
}

async function fetchPendingLabour(): Promise<LabourAd[]> {
  const data = await fetchJsonWithTimeout<{ items?: LabourAd[] }>(
    `${API_BASE_URL}/labour/pending`,
    { method: "GET" }
  );
  return data.items ?? [];
}

async function fetchPendingEquipment(): Promise<EquipmentAd[]> {
  const data = await fetchJsonWithTimeout<{ items?: EquipmentAd[] }>(
    `${API_BASE_URL}/equipment/pending`,
    { method: "GET" }
  );
  return data.items ?? [];
}

async function moderateLabour(
  id: string,
  action: "approve" | "reject"
): Promise<void> {
  await fetchJsonWithTimeout<{ ok?: boolean }>(
    `${API_BASE_URL}/labour/${encodeURIComponent(id)}/moderate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }
  );
}

async function moderateEquipment(
  id: string,
  action: "approve" | "reject"
): Promise<void> {
  await fetchJsonWithTimeout<{ ok?: boolean }>(
    `${API_BASE_URL}/equipment/${encodeURIComponent(id)}/moderate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }
  );
}

function money(value: number): string {
  return new Intl.NumberFormat("en-LK", {
    style: "currency",
    currency: "LKR",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

function isLabourOutlier(ad: LabourAd): boolean {
  return !!ad.price_outlier;
}

function isEquipmentOutlier(ad: EquipmentAd): boolean {
  return !!ad.hourly_price_outlier || !!ad.daily_price_outlier;
}

function normalizeQuery(q: string): string {
  return q.trim().toLowerCase();
}

function labourHaystack(ad: LabourAd): string {
  return [
    ad.id,
    ad.name,
    ad.location,
    ad.labour_type,
    ad.skill_level,
    ad.season ?? "",
    ad.crop_type ?? "",
    String(ad.hourly_rate ?? ""),
    String(ad.rating ?? ""),
    String(ad.jobs_completed ?? ""),
    String(ad.experience_years ?? ""),
    ad.available_day,
    ad.available_time,
    ad.price_outlier ? "price outlier" : "",
    ad.price_outlier_reason ?? "",
  ]
    .join(" ")
    .toLowerCase();
}

function equipmentHaystack(ad: EquipmentAd): string {
  return [
    ad.id,
    ad.equipment_type,
    ad.for_crop,
    ad.location,
    ad.nearest_major_district ?? "",
    ad.owner_name,
    ad.condition,
    String(ad.hourly_rate ?? ""),
    String(ad.daily_rate ?? ""),
    String(ad.rating ?? ""),
    String(ad.past_bookings ?? ""),
    ad.available_day,
    ad.available_time,
    ad.hourly_price_outlier ? "hourly outlier" : "",
    ad.daily_price_outlier ? "daily outlier" : "",
    ad.hourly_price_outlier_reason ?? "",
    ad.daily_price_outlier_reason ?? "",
  ]
    .join(" ")
    .toLowerCase();
}

function filterLabour(ads: LabourAd[], query: string): LabourAd[] {
  const n = normalizeQuery(query);
  if (!n) return ads;
  return ads.filter((ad) => labourHaystack(ad).includes(n));
}

function filterEquipment(ads: EquipmentAd[], query: string): EquipmentAd[] {
  const n = normalizeQuery(query);
  if (!n) return ads;
  return ads.filter((ad) => equipmentHaystack(ad).includes(n));
}

export default function App() {
  const [tab, setTab] = useState<Tab>("labour");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [labourAds, setLabourAds] = useState<LabourAd[]>([]);
  const [equipmentAds, setEquipmentAds] = useState<EquipmentAd[]>([]);
  const [pendingLabour, setPendingLabour] = useState<LabourAd[]>([]);
  const [pendingEquipment, setPendingEquipment] = useState<EquipmentAd[]>([]);
  const [actingKey, setActingKey] = useState("");
  const [searchLabour, setSearchLabour] = useState("");
  const [searchEquipment, setSearchEquipment] = useState("");
  const [searchPending, setSearchPending] = useState("");
  const [fraudModal, setFraudModal] = useState<{
    title: string;
    reasons: string[];
  } | null>(null);

  const filteredLabourAds = useMemo(
    () => filterLabour(labourAds, searchLabour),
    [labourAds, searchLabour]
  );
  const filteredEquipmentAds = useMemo(
    () => filterEquipment(equipmentAds, searchEquipment),
    [equipmentAds, searchEquipment]
  );
  const filteredPendingLabour = useMemo(
    () => filterLabour(pendingLabour, searchPending),
    [pendingLabour, searchPending]
  );
  const filteredPendingEquipment = useMemo(
    () => filterEquipment(pendingEquipment, searchPending),
    [pendingEquipment, searchPending]
  );

  const statLabel =
    tab === "pending" ? "Awaiting review" : "Listings on this page";
  const statTotal =
    tab === "pending"
      ? pendingLabour.length + pendingEquipment.length
      : tab === "labour"
        ? labourAds.length
        : equipmentAds.length;

  const statFiltered =
    tab === "pending"
      ? filteredPendingLabour.length + filteredPendingEquipment.length
      : tab === "labour"
        ? filteredLabourAds.length
        : filteredEquipmentAds.length;

  const searchActive =
    tab === "labour"
      ? normalizeQuery(searchLabour).length > 0
      : tab === "equipment"
        ? normalizeQuery(searchEquipment).length > 0
        : normalizeQuery(searchPending).length > 0;

  const title = useMemo(() => {
    if (tab === "labour") return "Labour Ads";
    if (tab === "equipment") return "Equipment Ads";
    return "Pending approval";
  }, [tab]);

  const subtitle = useMemo(
    () =>
      tab === "pending"
        ? "Newly posted ads are hidden from search until you approve them."
        : "Review approved labour and equipment listings (public search).",
    [tab]
  );

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        if (tab === "labour") {
          setLabourAds(await fetchLabourAds());
          return;
        }
        if (tab === "equipment") {
          setEquipmentAds(await fetchEquipmentAds());
          return;
        }
        const [l, e] = await Promise.all([
          fetchPendingLabour(),
          fetchPendingEquipment(),
        ]);
        setPendingLabour(l);
        setPendingEquipment(e);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to load ads";
        setError(msg);
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [tab]);

  async function handleModerate(
    kind: "labour" | "equipment",
    id: string,
    action: "approve" | "reject"
  ) {
    const key = `${kind}:${id}:${action}`;
    setActingKey(key);
    setError("");
    try {
      if (kind === "labour") {
        await moderateLabour(id, action);
        setPendingLabour((rows) => rows.filter((r) => r.id !== id));
      } else {
        await moderateEquipment(id, action);
        setPendingEquipment((rows) => rows.filter((r) => r.id !== id));
      }
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Could not update moderation";
      setError(msg);
    } finally {
      setActingKey("");
    }
  }

  function openFraudModalForLabour(ad: LabourAd) {
    const reasons = [ad.price_outlier_reason].filter(
      (v): v is string => !!v && v.trim().length > 0
    );
    if (!reasons.length) return;
    setFraudModal({
      title: `Labour fraud warning — ${ad.name || ad.id}`,
      reasons,
    });
  }

  function openFraudModalForEquipment(ad: EquipmentAd) {
    const reasons = [ad.hourly_price_outlier_reason, ad.daily_price_outlier_reason].filter(
      (v): v is string => !!v && v.trim().length > 0
    );
    if (!reasons.length) return;
    setFraudModal({
      title: `Equipment fraud warning — ${ad.equipment_type || ad.id}`,
      reasons,
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <div className="sidebar-brand">
          <span className="sidebar-logo">YieldSync</span>
          <span className="sidebar-sub">Admin</span>
        </div>
        <nav className="sidebar-nav">
          <button
            type="button"
            className={`sidebar-link${tab === "labour" ? " active" : ""}`}
            onClick={() => setTab("labour")}
            aria-current={tab === "labour" ? "page" : undefined}
          >
            Labour Ads
          </button>
          <button
            type="button"
            className={`sidebar-link${tab === "equipment" ? " active" : ""}`}
            onClick={() => setTab("equipment")}
            aria-current={tab === "equipment" ? "page" : undefined}
          >
            Equipment Ads
          </button>
          <button
            type="button"
            className={`sidebar-link${tab === "pending" ? " active" : ""}`}
            onClick={() => setTab("pending")}
            aria-current={tab === "pending" ? "page" : undefined}
          >
            Pending
          </button>
        </nav>
      </aside>

      <main className="main-area">
        <div className="container">
          <header className="header">
            <div>
              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>
            <div className="stats">
              <span>{searchActive ? "Matching filter" : statLabel}</span>
              <strong>{searchActive ? statFiltered : statTotal}</strong>
              {searchActive && (
                <span className="stats-note">of {statTotal} loaded</span>
              )}
            </div>
          </header>

          <section className="view-toggle" aria-label="Switch ad type">
            <button
              type="button"
              className={tab === "labour" ? "active" : ""}
              onClick={() => setTab("labour")}
            >
              Labour Ads
            </button>
            <button
              type="button"
              className={tab === "equipment" ? "active" : ""}
              onClick={() => setTab("equipment")}
            >
              Equipment Ads
            </button>
            <button
              type="button"
              className={tab === "pending" ? "active" : ""}
              onClick={() => setTab("pending")}
            >
              Pending
            </button>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>{title}</h2>
            </div>

            {!loading && !error && tab === "labour" && (
              <div className="search-bar">
                <label className="search-label" htmlFor="admin-search-labour">
                  Search labour ads
                </label>
                <input
                  id="admin-search-labour"
                  type="search"
                  className="search-input"
                  placeholder="Name, type, location, skill, id…"
                  value={searchLabour}
                  onChange={(e) => setSearchLabour(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
            )}

            {!loading && !error && tab === "equipment" && (
              <div className="search-bar">
                <label className="search-label" htmlFor="admin-search-equipment">
                  Search equipment ads
                </label>
                <input
                  id="admin-search-equipment"
                  type="search"
                  className="search-input"
                  placeholder="Type, crop, owner, location, id…"
                  value={searchEquipment}
                  onChange={(e) => setSearchEquipment(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
            )}

            {!loading && !error && tab === "pending" && (
              <div className="search-bar">
                <label className="search-label" htmlFor="admin-search-pending">
                  Search pending ads
                </label>
                <input
                  id="admin-search-pending"
                  type="search"
                  className="search-input"
                  placeholder="Filters both labour and equipment pending lists…"
                  value={searchPending}
                  onChange={(e) => setSearchPending(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
            )}

            {loading && (
              <p className="status">
                {tab === "pending"
                  ? "Loading pending ads…"
                  : `Loading ${title.toLowerCase()}…`}
              </p>
            )}
            {error && <p className="status error">{error}</p>}

            {!loading && !error && tab === "labour" && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Location</th>
                      <th>Rate</th>
                      <th>Rating</th>
                      <th>Jobs</th>
                      <th>Experience</th>
                      <th>Availability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLabourAds.map((ad) => (
                      <tr key={ad.id}>
                        <td>{ad.name || "-"}</td>
                        <td>{ad.labour_type || "-"}</td>
                        <td>{ad.location || "-"}</td>
                        <td>{money(ad.hourly_rate)}/hr</td>
                        <td>{ad.rating?.toFixed(1) ?? "0.0"}</td>
                        <td>{ad.jobs_completed ?? 0}</td>
                        <td>{ad.experience_years ?? 0} yrs</td>
                        <td>
                          {ad.available_day || "-"} | {ad.available_time || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!labourAds.length && (
                  <p className="empty">No labour ads found.</p>
                )}
                {!!labourAds.length && !filteredLabourAds.length && (
                  <p className="empty">No rows match your search.</p>
                )}
              </div>
            )}

            {!loading && !error && tab === "equipment" && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Type</th>
                      <th>Crop</th>
                      <th>Location</th>
                      <th>Hourly</th>
                      <th>Daily</th>
                      <th>Rating</th>
                      <th>Bookings</th>
                      <th>Owner</th>
                      <th>Condition</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredEquipmentAds.map((ad) => (
                      <tr key={ad.id}>
                        <td>{ad.equipment_type || "-"}</td>
                        <td>{ad.for_crop || "-"}</td>
                        <td>{ad.location || "-"}</td>
                        <td>{money(ad.hourly_rate)}</td>
                        <td>{money(ad.daily_rate)}</td>
                        <td>{ad.rating?.toFixed(1) ?? "0.0"}</td>
                        <td>{ad.past_bookings ?? 0}</td>
                        <td>{ad.owner_name || "-"}</td>
                        <td>{ad.condition || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!equipmentAds.length && (
                  <p className="empty">No equipment ads found.</p>
                )}
                {!!equipmentAds.length && !filteredEquipmentAds.length && (
                  <p className="empty">No rows match your search.</p>
                )}
              </div>
            )}

            {!loading && !error && tab === "pending" && (
              <div className="pending-sections">
                <div className="pending-block">
                  <h3 className="pending-heading">Labour — pending</h3>
                  <div className="table-wrap">
                    <table className="pending-table">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Type</th>
                          <th>Location</th>
                          <th>Rate</th>
                          <th>Rating</th>
                          <th>Availability</th>
                          <th>Fraud risk</th>
                          <th className="th-actions">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPendingLabour.map((ad) => (
                          <tr
                            key={ad.id}
                            className={isLabourOutlier(ad) ? "warning-row clickable-row" : ""}
                            onClick={() => openFraudModalForLabour(ad)}
                          >
                            <td>{ad.name || "-"}</td>
                            <td>{ad.labour_type || "-"}</td>
                            <td>{ad.location || "-"}</td>
                            <td>{money(ad.hourly_rate)}/hr</td>
                            <td>{ad.rating?.toFixed(1) ?? "0.0"}</td>
                            <td>
                              {ad.available_day || "-"} |{" "}
                              {ad.available_time || "-"}
                            </td>
                            <td>
                              <span
                                className={`risk-badge${isLabourOutlier(ad) ? " high" : " low"}`}
                              >
                                {isLabourOutlier(ad) ? "Price outlier" : "Normal"}
                              </span>
                            </td>
                            <td>
                              <div className="action-cell">
                                <button
                                  type="button"
                                  className="btn-approve"
                                  disabled={
                                    !!actingKey &&
                                    actingKey.startsWith(`labour:${ad.id}:`)
                                  }
                                  onClick={(e) =>
                                    void (async () => {
                                      e.stopPropagation();
                                      await handleModerate("labour", ad.id, "approve");
                                    })()
                                  }
                                >
                                  Approve
                                </button>
                                <button
                                  type="button"
                                  className="btn-reject"
                                  disabled={
                                    !!actingKey &&
                                    actingKey.startsWith(`labour:${ad.id}:`)
                                  }
                                  onClick={(e) =>
                                    void (async () => {
                                      e.stopPropagation();
                                      await handleModerate("labour", ad.id, "reject");
                                    })()
                                  }
                                >
                                  Reject
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {!pendingLabour.length && (
                      <p className="empty">No labour ads awaiting review.</p>
                    )}
                    {!!pendingLabour.length && !filteredPendingLabour.length && (
                      <p className="empty">No labour rows match your search.</p>
                    )}
                  </div>
                </div>

                <div className="pending-block">
                  <h3 className="pending-heading">Equipment — pending</h3>
                  <div className="table-wrap">
                    <table className="pending-table">
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Crop</th>
                          <th>Location</th>
                          <th>Hourly</th>
                          <th>Daily</th>
                          <th>Owner</th>
                          <th>Condition</th>
                          <th>Fraud risk</th>
                          <th className="th-actions">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPendingEquipment.map((ad) => (
                          <tr
                            key={ad.id}
                            className={isEquipmentOutlier(ad) ? "warning-row clickable-row" : ""}
                            onClick={() => openFraudModalForEquipment(ad)}
                          >
                            <td>{ad.equipment_type || "-"}</td>
                            <td>{ad.for_crop || "-"}</td>
                            <td>{ad.location || "-"}</td>
                            <td>{money(ad.hourly_rate)}</td>
                            <td>{money(ad.daily_rate)}</td>
                            <td>{ad.owner_name || "-"}</td>
                            <td>{ad.condition || "-"}</td>
                            <td>
                              <span
                                className={`risk-badge${isEquipmentOutlier(ad) ? " high" : " low"}`}
                              >
                                {isEquipmentOutlier(ad) ? "Price outlier" : "Normal"}
                              </span>
                            </td>
                            <td>
                              <div className="action-cell">
                                <button
                                  type="button"
                                  className="btn-approve"
                                  disabled={
                                    !!actingKey &&
                                    actingKey.startsWith(`equipment:${ad.id}:`)
                                  }
                                  onClick={(e) =>
                                    void (async () => {
                                      e.stopPropagation();
                                      await handleModerate("equipment", ad.id, "approve");
                                    })()
                                  }
                                >
                                  Approve
                                </button>
                                <button
                                  type="button"
                                  className="btn-reject"
                                  disabled={
                                    !!actingKey &&
                                    actingKey.startsWith(`equipment:${ad.id}:`)
                                  }
                                  onClick={(e) =>
                                    void (async () => {
                                      e.stopPropagation();
                                      await handleModerate("equipment", ad.id, "reject");
                                    })()
                                  }
                                >
                                  Reject
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {!pendingEquipment.length && (
                      <p className="empty">
                        No equipment ads awaiting review.
                      </p>
                    )}
                    {!!pendingEquipment.length && !filteredPendingEquipment.length && (
                      <p className="empty">
                        No equipment rows match your search.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
      {fraudModal && (
        <div className="modal-backdrop" onClick={() => setFraudModal(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>{fraudModal.title}</h3>
            <p className="modal-subtitle">Fraud detection reasons</p>
            <ul className="modal-reason-list">
              {fraudModal.reasons.map((reason, idx) => (
                <li key={`${idx}-${reason}`}>{reason}</li>
              ))}
            </ul>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-approve"
                onClick={() => setFraudModal(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
