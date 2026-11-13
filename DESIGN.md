# ShopSense AI — Design System & UI Specification

This document defines the visual language, tokens, and component standards for **ShopSense AI**. All frontend components in `frontend/src/` must follow these specifications.

---

## 1. Color System — **Midnight Obsidian**

### 1.1 Surface & Neutral Hierarchy
| Token | Hex | Usage |
| :--- | :--- | :--- |
| `--bg-canvas` | `#05070A` | Primary application backdrop |
| `--bg-surface` | `#0D1117` | Cards, chart containers, data tables |
| `--bg-elevated` | `#161B22` | Interactive controls, sliders, hover states, formula blocks |
| `--border-subtle` | `#30363D` | 1px structural dividers & card outlines |
| `--text-primary` | `#F0F6FC` | Headings, KPI values, active labels |
| `--text-secondary` | `#8B949E` | Descriptions, axis labels, table headers |
| `--text-muted` | `#6E7681` | Helper captions, footnotes, timestamps |

### 1.2 ML Paradigm Identity Accents
Each tab and module uses a dedicated accent for instant visual orientation during the live demo:
| Module | Owner | Accent Name | Hex |
| :--- | :--- | :--- | :--- |
| **1. Data Pipeline** | Member 1 | Electric Cyan | `#22D3EE` |
| **2. Unsupervised (K-Means)** | Member 2 | Mint Emerald | `#34D399` |
| **3. Supervised (Random Forest)** | Member 3 | Soft Indigo | `#818CF8` |
| **4. Reinforcement (Q-Learning)** | Member 4 | Warm Amber | `#FBBF24` |

### 1.3 Semantic Status Colors
- **Success / Retained / Optimal:** `#34D399`
- **Warning / Medium Risk:** `#FBBF24`
- **Danger / High Churn Risk / Error:** `#FB7185`

---

## 2. Typography — **Geometric Tech**

### 2.1 Font Families
- **Display & Headings (`--font-display`):** `Sora` (Weights: `600`, `700`, `800`) — tight tracking (`-0.025em`) for sharp technical authority.
- **Body & UI (`--font-body`):** `Manrope` (Weights: `400`, `500`, `600`) — high legibility at small UI sizes.
- **Monospace, Metrics & Formulas (`--font-mono`):** `Source Code Pro` (Weights: `400`, `500`, `700`) with `font-variant-numeric: tabular-nums` so numbers never jitter when sliders move.

### 2.2 Type Scale
| Role | Font | Weight | Size | Tracking |
| :--- | :--- | :--- | :--- | :--- |
| **Page Title (H1)** | `Sora` | `800` | `32px` | `-0.03em` |
| **Section Title (H2)** | `Sora` | `700` | `22px` | `-0.02em` |
| **Card Title (H3)** | `Sora` | `700` | `16px` | `-0.01em` |
| **Body Copy** | `Manrope` | `400` | `14px` | `0` |
| **UI Label / Pill** | `Manrope` | `600` | `13px` | `0` |
| **KPI Metric** | `Source Code Pro` | `700` | `30px` | `-0.02em` |
| **Formula / Code** | `Source Code Pro` | `500` | `14px` | `0` |

---

## 3. Layout & Navigation — **Left Sidebar + Split-Pane Workbench (Option 2)**

### 3.1 Application Shell
- **Left Navigation Rail (`240px` fixed on `lg+`, top bar on mobile):**
  - Brand mark (`ShopSense AI`) + subtitle.
  - 4 Paradigm Tabs with Member ownership pills (`Member 1`–`Member 4`) and active paradigm accent highlight.
  - Bottom status card showing FastAPI connection health and active dataset row count.
- **Main Content Area:**
  - Full-height `#05070A` canvas with `24px` padding.

### 3.2 Split-Pane Workbench (Per Paradigm)
- **Left Drawer (`300px`–`320px`, sticky on `lg+`):**
  - Hyperparameter sliders & algorithm selectors.
  - Primary Run / Train CTA button styled with the module's signature accent.
  - **Algorithm & Code Spotlight** card showing the exact formula (e.g., Bellman update, Gini impurity, Euclidean standardization) and source file path (`ml/*.py`) for live Q&A.
- **Right Visual Canvas (`1fr`):**
  - Top 3–4 column KPI metric row (`Source Code Pro` `tabular-nums`).
  - Side-by-side interactive Recharts visualizations.
  - Bottom interactive simulator, persona cards, or 24-state Q-Table.

