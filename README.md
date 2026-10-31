<div align="center">

# 💊 MediGuide AI
### *Next-Generation Intelligent Pharmaceutical Assistant, Formulary RAG Engine & CDSCO Government Dataset Synchronizer*

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38Bdf8?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Firebase Firestore](https://img.shields.io/badge/Firebase_Firestore-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

*An enterprise-grade, ChatGPT-class clinical and pharmaceutical assistant built to bridge the gap between patient care, local drug formulary verification, package OCR scanning, drug interaction analysis, and automated Indian government (CDSCO) dataset synchronization.*

</div>

---

## 🌟 Executive Overview

**MediGuide AI** is a production-grade healthcare and pharmaceutical intelligence platform designed to empower patients, clinicians, and hospital administrators. It features a comprehensive **253,973-record verified Indian medicine catalog**, multimodal package scanning (combining Gemini Vision AI and Tesseract.js OCR), multi-turn conversational AI with RAG grounding, real-time drug interaction safety checks, and an **Automatic Government Dataset Synchronization System** that syncs official CDSCO approvals every 10 days.

---

## 🚀 Key Features & Capabilities

### 1. 📚 253,973-Record Verified Indian Medicine Formulary
- Complete coverage of Indian pharmaceutical brands, active ingredients, strengths, dosage forms, manufacturer details, pack sizes, and MRP pricing.
- Blazing-fast multi-index search supporting brand names, salt compositions, and therapeutic classes.

### 2. 💬 ChatGPT-Grade Multimodal Ask MediGuide AI
- **Conversational Intelligence**: Natural multi-turn dialogue with memory, context-aware follow-up pronoun resolution ("What about its side effects?"), and intelligent clarification prompting.
- **Multilingual Support**: Fluent conversational reasoning and RAG answers in **English, Telugu (తెలుగు), Hindi (हिन्दी), and Tamil (தமிழ்)** while strictly preserving medicine brand names, active ingredients, and dosage units in standard script for patient safety.
- **Voice Agent Integration**: Built-in speech synthesis (TTS) powered by Gemini Cloud audio models.

### 3. 📷 Hybrid OCR Package & Blister Strip Scanner
- Instant optical character recognition (OCR) leveraging pre-warmed **Tesseract.js** and **Gemini 3.1 Flash Vision** for blister foil strips, medicine bottles, and prescription sheets.
- Automatically extracts brand names, active ingredients, batch numbers, manufacturing dates, expiry dates, and MRP pricing.

### 4. 🔄 Automatic Government Dataset Synchronization System
- **Scheduled Sync**: Automatically checks official CDSCO portals (CDSCO Approved Drugs, Approved New Drugs, FDC & Subsequent Approvals, and Safety Notices) every 10 days by default.
- **Configurable Frequencies**: Super Admins can configure synchronization to run every 10 days, monthly, or on-demand via a manual **Sync Now** button.
- **Publishing Review Queue**: Two-stage validation pipeline that automatically filters duplicates and sends uncertain or partial records to a Super Admin review queue before public RAG indexing.

### 5. ⚠️ Advanced Drug Interaction & Comparison Centre
- **Pairwise Interaction Matrix**: Real-time clinical safety checks across multiple medicines displaying severity levels (*Contraindicated, Serious, Use with Caution, No Major Interaction*).
- **Side-by-Side Comparison**: Side-by-side pharmacological comparison of active ingredients, pricing, indications, and non-interchangeability safety notices.

### 6. 🔐 Secure Role-Based Admin & Super Admin Consoles
- **Super Admin Governance**: Full administrative control with isolated Super Admin credentials (`ramcharanreddykaitha@gmail.com` / `Ramcharan@70`), audit logs, system health monitoring, and dataset sync controls.
- **Clinical Admin Console**: Operational management of user accounts, feedback queues, and customer support queries routed from website contact forms.

---

## 🏗️ Architecture & Tech Stack

### 🔄 Automated Dataset Synchronization & RAG Pipeline

<div align="center">
  <img src="/public/pipeline_architecture.svg" alt="MediGuide AI Pipeline Architecture" width="100%" style="border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;" />
</div>

```
┌────────────────────────────────────────────────────────┐
│                      Client Layer                      │
│     React 18 • Vite • Tailwind CSS • Lucide Icons      │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / REST & Bearer Tokens
┌──────────────────────────▼─────────────────────────────┐
│                    Backend API Server                  │
│       Node.js • Express • Vite Middleware (Port 3000)   │
└──────┬───────────────────┬───────────────────┬─────────┘
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼─────────┐
│ Google GenAI│     │ TesseractJS │     │ Cloud Firestore│
│  SDK (RAG)  │     │   Vision    │     │ & Local JSON   │
└─────────────┘     └─────────────┘     └────────────────┘
```

---

## 🛠️ Project Structure

```
mediguide-ai/
├── server.ts                       # Express backend, authentication, RAG engine, & CDSCO sync scheduler
├── data/                           # Persistent JSON storage (auth accounts, sessions, queries, sync state)
├── public/                         # Static assets, medicine JSON catalogs & icons
├── src/
│   ├── App.tsx                     # Main React application & routing container
│   ├── main.tsx                    # React entry point
│   ├── components/                 # Modular UI views
│   │   ├── PublicWebsite.tsx       # Landing page, sign-in, register & contact enquiry forms
│   │   ├── DashboardAndChatViews.tsx # Ask MediGuide AI chat & personal dashboard
│   │   ├── ScanAndCompareViews.tsx # OCR package scanner & drug comparison/interaction centre
│   │   ├── PersonalAndAdminViews.tsx # My medicines, reminders, Admin Console & Super Admin portal
│   │   ├── MedicinesCatalogView.tsx  # 253K+ medicine formulary directory browser
│   │   ├── DatabaseSchemaView.tsx    # Database architecture & collection browser
│   │   └── LanguageSelector.tsx      # Multilingual language switcher
│   └── lib/
│       ├── mediguideEngine.ts      # Core pharmaceutical RAG, OCR parser, and safety guard engine
│       ├── mediguideEngine.test.ts # Comprehensive test suite (38 automated unit tests)
│       ├── firebaseClient.ts       # Firebase Firestore & Auth client integration
│       └── i18n.ts                 # i18next internationalization setup (English, Telugu, Hindi)
└── package.json                    # Dependencies & scripts
```

---

## ⚙️ Getting Started & Local Development

### Prerequisites
- Node.js (v18+ recommended)
- npm or yarn

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/mediguide-ai.git
cd mediguide-ai
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
PORT=3000
GEMINI_API_KEY=your_google_gemini_api_key_here
ADMIN_EMAIL=admin@mediguide.ai
ADMIN_PASSWORD=Admin@MediGuide#2026
```

### 4. Run the Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 5. Run Automated Tests
```bash
npx tsx --test src/lib/mediguideEngine.test.ts
```

---

## 🔑 Default Authentication Credentials

| Role | Email | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Super Admin** | `ramcharanreddykaitha@gmail.com` | `Ramcharan@70` | Full System Governance, Dataset Sync & User Control |
| **Clinical Admin** | `admin@mediguide.ai` | `Admin@MediGuide#2026` | Admin Console, Query Resolution & Formulary Oversight |
| **Patient / User** | *Self-Register via Webpage* | *Custom 8+ Chars* | Personal Workspace, Scan, Chat, Favourites & Reminders |

---

## 🛡️ Safety & Compliance Disclaimer

MediGuide AI is designed as a clinical and pharmaceutical assistant to provide evidence-based RAG retrieval from verified datasets and official CDSCO approval notices. It does not replace professional medical advice, diagnosis, or prescription. Always consult a qualified healthcare provider before altering medication regimens.

---

## 📄 License

Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for more information.

<div align="center">
  <p>Built with precision for healthcare and pharmaceutical excellence.</p>
</div>
