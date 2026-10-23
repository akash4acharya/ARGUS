# 01 — DESIGN SYSTEM: CCTV Speed & Telemetry Overlay Suite

## 1. Core Visual Philosophy
Operators watching pre-existing CCTV grids suffer from visual fatigue caused by cluttered screens. This system follows an **Exception-First Visual Hierarchy**: normal traffic blends quietly into the background with muted styling, while vehicles approaching or exceeding the speed limit trigger high-contrast visual alerts.

---

## 2. Semantic Color Tokens
All colors are specified in both **HEX (for Web UI)** and **BGR Tuple (for OpenCV drawing)** because OpenCV stores color channels in Blue-Green-Red order rather than Red-Green-Blue.

| State / Element | Meaning | Web HEX | OpenCV BGR Tuple | Opacity / Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Surface Primary** | Main dashboard background | `#0B0F17` | `(23, 15, 11)` | 100% solid dark slate |
| **Surface Elevated** | Violation cards & sidebars | `#161E2E` | `(46, 30, 22)` | 100% solid |
| **Status: Nominal** | Speed < 85% of limit | `#10B981` | `(129, 185, 16)` | Thin 1.5px bounding box |
| **Status: Warning** | Speed 85%–99% of limit | `#F59E0B` | `(11, 158, 245)` | 2px box + amber speed tag |
| **Status: Violation** | Speed >= 100% of limit | `#EF4444` | `(68, 68, 239)` | 3px box + flashing red header |
| **Calibration Zone** | 4-point Homography polygon | `#38BDF8` | `(248, 189, 56)` | 20% fill alpha, 2px dashed border |
| **Tire Contact Point** | Bottom-center ground anchor | `#F43F5E` | `(94, 63, 244)` | 5px solid circle on road plane |

---

## 3. OpenCV On-Screen Display (HUD) Geometry Rules
Never draw full opaque boxes over vehicles, as this hides license plates and vehicle features.

1. **Corner-Bracket Bounding Boxes:** Draw only the 4 corners (length = 20% of box width/height) for nominal vehicles; switch to a full solid rectangle only when a vehicle enters `Warning` or `Violation` state.
2. **Ground-Plane Anchor Dot:** Always render a `5px` dot at `(x_center, y_bottom)` of the bounding box. This visually confirms for the operator where the math engine thinks the tires touch the asphalt.
3. **Floating Telemetry Pill:** Above each tracked vehicle, render a dark translucent rectangle (`alpha = 0.75`) containing fixed-width text:
   - Format: `#[ID] | [CLASS] | [SPEED] km/h`
   - Example: `#042 | CAR | 46.2 km/h`
4. **Calibration Polygon Overlay:** Render the 4-point homography region as a semi-transparent cyan trapezoid on the road so the operator can immediately spot if physical camera sway has misaligned the calibration zone.

---

## 4. Typography & Monospace Enforcement
Numbers that change 30 times a second will jitter left and right unless every digit occupies the exact same horizontal pixel width—a property known as **tabular monospace rendering**.

- **Primary UI Font (Headers & Labels):** `Inter` or `system-ui`, weights `500` (Medium) and `700` (Bold).
- **Telemetry & Speed Readouts (Mandatory Monospace):** `JetBrains Mono` or `Fira Code`. Every speed readout, timestamp, FPS counter, and vehicle ID must use monospace formatting.
- **OpenCV Font:** `cv2.FONT_HERSHEY_SIMPLEX` with `LINE_AA` (anti-aliased edges) scaled dynamically to frame height (`scale = frame_height / 1400`).

---

## 5. Operator Dashboard Layout (3-Zone Grid)
The web interface monitoring the pre-existing CCTV feed uses a fixed, non-scrolling `100vh` viewport divided into three functional zones:

- **Zone A — Top Telemetry Bar (`64px` height):** Displays active camera stream status, real-time processing FPS, active frame-time delta ($\Delta t$ in ms), configured speed limit threshold, and a "Recalibrate 4-Point Grid" toggle button.
- **Zone B — Primary Video Canvas (`70%` width, left):** Renders the live annotated video feed and supports interactive mouse clicks to drag the 4 calibration anchor points $(P_1, P_2, P_3, P_4)$ directly over the road.
- **Zone C — Live Violation Audit Feed (`30%` width, right):** Real-time scrolling stack of violation cards pushed via WebSockets. Each card displays:
  - Cropped vehicle snapshot
  - Peak recorded speed in bold red (`54.8 km/h` in a `30 km/h` zone)
  - Track ID, Vehicle Class, and ISO-8601 Timestamp
