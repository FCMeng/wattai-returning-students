(function () {
  "use strict";

  const FORM_URL = "https://clemson.ca1.qualtrics.com/jfe/form/SV_3xQLX0ZmyK86AFo";
  const TERM = "2026-Fall";
  const script = document.currentScript;

  function cleanPageUrl(value) {
    const url = new URL(value || window.location.href, document.baseURI);
    url.hash = "";
    url.search = "";
    return url.href;
  }

  function feedbackUrl(element) {
    const url = new URL(FORM_URL);
    const title = element.dataset.pageTitle || document.querySelector("h1")?.textContent.trim() || document.title;
    url.searchParams.set("material_id", element.dataset.materialId || "course-page");
    url.searchParams.set("page_title", title);
    url.searchParams.set("page_url", cleanPageUrl(element.dataset.pageUrl));
    url.searchParams.set("content_type", element.dataset.contentType || "course-page");
    url.searchParams.set("term", TERM);
    return url.href;
  }

  function configureLink(link) {
    link.href = feedbackUrl(link);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }

  document.querySelectorAll("[data-feedback-link]").forEach(configureLink);

  if (script?.dataset.autoButton === "true") {
    const style = document.createElement("style");
    style.textContent = `
      .material-feedback-float {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 10000;
        max-width: 300px;
        padding: 12px 14px;
        color: #172638;
        background: rgba(255,255,255,.97);
        border: 1px solid #d8e0e8;
        border-left: 5px solid #b85c18;
        border-radius: 10px;
        box-shadow: 0 8px 25px rgba(23,38,56,.18);
        font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .material-feedback-float a { color: #174f82; font-size: 16px; font-weight: 750; }
      .material-feedback-float span { display: block; margin-top: 4px; }
      @media (max-width: 640px) {
        .material-feedback-float { right: 10px; bottom: 10px; left: 10px; max-width: none; }
      }
    `;
    document.head.appendChild(style);

    const panel = document.createElement("aside");
    panel.className = "material-feedback-float";
    panel.setAttribute("aria-label", "Material feedback");
    const link = document.createElement("a");
    link.textContent = "Share feedback";
    link.dataset.materialId = script.dataset.materialId || "course-page";
    link.dataset.contentType = script.dataset.contentType || "course-page";
    link.dataset.pageTitle = script.dataset.pageTitle || document.title;
    configureLink(link);
    const help = document.createElement("span");
    help.textContent = "Tell us what helped, what was unclear, or what broke.";
    panel.append(link, help);
    document.body.appendChild(panel);
  }
})();
