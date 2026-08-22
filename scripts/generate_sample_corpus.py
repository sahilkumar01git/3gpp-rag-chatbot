"""
Generates the bundled SAMPLE PDF corpus used for demos, the CLI/API
quick-start, and the evaluation harness.

IMPORTANT — copyright note
---------------------------
These five PDFs are 100% original content written for this project. They
mimic the *structure* of real 3GPP specifications (running headers with a
spec number/version, numbered clause headings, tables) so the ingestion
pipeline has something realistic to parse — but the wording is not copied
from any actual 3GPP document. This sidesteps redistributing copyrighted
standards text while still giving you a genuine multi-document PDF corpus
to exercise the full pipeline end to end.

For real usage, download the actual specifications you need from
https://www.3gpp.org/specifications-technologies (free, no login) and
drop the PDFs into `data/raw_pdfs/` — the ingestion pipeline (pdf_parser.py
+ chunker.py) works identically on real 3GPP PDFs; nothing needs to change.

Run:
    python scripts/generate_sample_corpus.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs" / "sample_corpus"

styles = getSampleStyleSheet()
h1 = ParagraphStyle("ClauseH1", parent=styles["Heading1"], fontSize=13, spaceAfter=8, spaceBefore=14)
h2 = ParagraphStyle("ClauseH2", parent=styles["Heading2"], fontSize=11.5, spaceAfter=6, spaceBefore=10)
body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=6)
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=16, spaceAfter=10)
disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"], fontSize=8, textColor=colors.grey, spaceAfter=4)


def _header_footer(spec_id: str, version: str, release: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(2 * cm, A4[1] - 1.3 * cm, f"3GPP {spec_id} version {version} Release {release}")
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, "SYNTHETIC SAMPLE — not an official 3GPP document")
        canvas.drawString(2 * cm, 1.2 * cm, f"3GPP {spec_id}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _make_table(header: list[str], rows: list[list[str]]) -> Table:
    data = [header] + rows
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _clause(number: str, title: str, paragraphs: list[str], level: int = 1):
    style = h1 if level == 1 else h2
    story = [Paragraph(f"{number}\u2002{title}", style)]
    for p in paragraphs:
        story.append(Paragraph(p, body))
    return story


def _build_pdf(filename: str, spec_id: str, version: str, release: str, title: str, story_body: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        topMargin=2.2 * cm,
        bottomMargin=2.0 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    story = [
        Paragraph(f"3GPP {spec_id}", title_style),
        Paragraph(title, styles["Heading2"]),
        Paragraph(f"Version {version} &nbsp;&nbsp; Release {release}", body),
        Paragraph(
            "SYNTHETIC SAMPLE DOCUMENT — written for demonstration and testing of a retrieval-augmented "
            "chatbot. This is original content structured to resemble a 3GPP technical specification; it "
            "is NOT an official 3GPP publication and must not be used as a source of normative requirements.",
            disclaimer_style,
        ),
        Spacer(1, 0.4 * cm),
    ]
    story.extend(story_body)

    draw_fn = _header_footer(spec_id, version, release)
    doc.build(story, onFirstPage=draw_fn, onLaterPages=draw_fn)
    print(f"Wrote {path}")


# ─────────────────────────────────────────────────────────────────────────
# 1. TS 38.331 — NR Radio Resource Control (RRC) protocol
# ─────────────────────────────────────────────────────────────────────────

def build_ts_38331():
    story = []
    story += _clause(
        "4.2.1", "RRC states",
        [
            "A UE in NR is, at any point in time, in one of three RRC states: RRC_IDLE, RRC_INACTIVE, or "
            "RRC_CONNECTED. In RRC_IDLE, the UE monitors paging and performs cell (re)selection but has no "
            "established RRC connection with any gNB. In RRC_INACTIVE, the UE retains its AS context and can "
            "resume the connection with reduced signalling overhead. In RRC_CONNECTED, a full RRC connection "
            "is established between the UE and the serving gNB, and the UE reports channel quality and "
            "mobility measurements as configured.",
        ],
    )
    story += _clause(
        "5.3.3", "RRC connection establishment",
        [
            "RRC connection establishment is triggered by the UE for reasons including mobile originating "
            "data, mobile originating signalling, or a response to paging. The UE initiates the procedure by "
            "sending an RRCSetupRequest message on the common control channel. Upon receiving RRCSetupRequest, "
            "the gNB replies with RRCSetup, which configures Signalling Radio Bearer 1 (SRB1). The UE shall "
            "respond with RRCSetupComplete, after which the connection is considered established.",
            "If the network is unable to admit the request, it shall respond with RRCReject and may include a "
            "wait timer instructing the UE to refrain from re-attempting for a specified duration.",
        ],
    )
    story += _clause(
        "5.3.13", "RRC connection re-establishment",
        [
            "The UE shall initiate RRC connection re-establishment upon detecting radio link failure, upon "
            "handover failure, upon mobility from NR failure, upon integrity check failure, or upon an RRC "
            "connection reconfiguration failure, provided the UE has a stored AS security context and a "
            "suitable cell is found within the re-establishment timer T311.",
        ],
    )
    story.append(PageBreak())
    story += _clause(
        "6.3.2", "Timers and constants (RRC)",
        [
            "The following timers govern RRC procedures. T300 bounds how long the UE waits for RRCSetup or "
            "RRCReject after sending RRCSetupRequest. T310 supervises detection of radio link failure "
            "following N310 consecutive out-of-sync indications. T311 bounds the UE's search for a suitable "
            "cell during re-establishment.",
        ],
    )
    story.append(
        _make_table(
            ["Timer", "Default value", "Started when", "Stopped when"],
            [
                ["T300", "1000 ms", "RRCSetupRequest is sent", "RRCSetup or RRCReject is received"],
                ["T310", "2000 ms", "N310 out-of-sync indications received", "N311 in-sync indications received"],
                ["T311", "3000 ms", "Re-establishment procedure is initiated", "A suitable cell is selected"],
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story += _clause(
        "6.3.5", "Maximum values (RRC)",
        [
            "N310, the number of consecutive out-of-sync indications required before starting T310, and N311, "
            "the number of consecutive in-sync indications required to stop it, are both configurable by the "
            "network within the ranges below.",
        ],
    )
    story.append(
        _make_table(
            ["Constant", "Range", "Typical network setting"],
            [["N310", "1 to 20", "1"], ["N311", "1 to 10", "1"]],
        )
    )

    _build_pdf("ts_38331_rrc_sample.pdf", "TS 38.331", "17.6.0", "17", "Radio Resource Control (RRC) protocol specification", story)


# ─────────────────────────────────────────────────────────────────────────
# 2. TS 23.501 — System architecture for the 5G System
# ─────────────────────────────────────────────────────────────────────────

def build_ts_23501():
    story = []
    story += _clause(
        "4.2.1", "Network functions overview",
        [
            "The 5G Core network is built from a set of network functions (NFs) that interact through a "
            "service-based architecture. The Access and Mobility Management Function (AMF) terminates the "
            "NAS signalling from the UE and handles registration, connection, and mobility management. The "
            "Session Management Function (SMF) is responsible for session establishment, IP address "
            "allocation, and selection of the User Plane Function (UPF) that handles user data forwarding.",
        ],
    )
    story += _clause(
        "4.2.3", "Policy and data management functions",
        [
            "The Policy Control Function (PCF) provides policy rules to control-plane functions such as the "
            "AMF and SMF, including QoS and charging-related policies. The Unified Data Management (UDM) "
            "function stores subscription data and generates authentication credentials in cooperation with "
            "the Authentication Server Function (AUSF). The Network Repository Function (NRF) maintains a "
            "registry of available NF instances and supports service discovery between them.",
        ],
    )
    story += _clause(
        "5.15.3", "Network slicing",
        [
            "Network slicing allows the 5G System to support multiple logical networks, each tailored to a "
            "particular set of requirements, over a shared physical infrastructure. A Network Slice is "
            "identified by Single Network Slice Selection Assistance Information (S-NSSAI), which comprises a "
            "Slice/Service Type (SST) and an optional Slice Differentiator (SD). A UE may be simultaneously "
            "served by up to eight Network Slices via a single registration.",
        ],
    )
    story.append(PageBreak())
    story += _clause(
        "5.15.4", "Standardized SST values",
        [
            "3GPP defines a small set of standardized Slice/Service Type values so that operators can achieve "
            "baseline slice interoperability without prior agreement.",
        ],
    )
    story.append(
        _make_table(
            ["SST value", "Slice/Service type", "Characteristics"],
            [
                ["1", "eMBB", "Enhanced Mobile Broadband — optimised for high data rates"],
                ["2", "URLLC", "Ultra-Reliable Low-Latency Communications"],
                ["3", "MIoT", "Massive IoT — optimised for high device density, low data rate"],
                ["4", "V2X", "Vehicle-to-Everything communications"],
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story += _clause(
        "8.2.1", "Registration management states",
        [
            "From a registration management perspective, a UE is in one of two states with respect to the "
            "AMF: RM-DEREGISTERED, in which the UE is not registered with the network and the AMF holds no "
            "valid location or routing information for it, and RM-REGISTERED, in which the UE has successfully "
            "completed a registration procedure and the network can reach it for paging or session-related "
            "signalling.",
        ],
    )

    _build_pdf("ts_23501_architecture_sample.pdf", "TS 23.501", "17.9.0", "17", "System architecture for the 5G System (5GS)", story)


# ─────────────────────────────────────────────────────────────────────────
# 3. TS 24.501 — Non-Access-Stratum (NAS) protocol for 5GS
# ─────────────────────────────────────────────────────────────────────────

def build_ts_24501():
    story = []
    story += _clause(
        "5.1.3", "Registration procedure",
        [
            "The registration procedure is used by the UE to register with the network for initial "
            "registration, mobility registration update, periodic registration update, or emergency "
            "registration. The UE sends a REGISTRATION REQUEST message to the AMF. If authentication and "
            "security procedures complete successfully and the AMF accepts the request, it responds with a "
            "REGISTRATION ACCEPT message, which the UE acknowledges with REGISTRATION COMPLETE if new 5G-GUTI "
            "or TAI list values were assigned.",
        ],
    )
    story += _clause(
        "5.5.1", "De-registration procedure",
        [
            "De-registration may be UE-initiated or network-initiated. In UE-initiated de-registration, the UE "
            "sends a DEREGISTRATION REQUEST indicating whether it wishes to switch off. In network-initiated "
            "de-registration, the AMF sends a DEREGISTRATION REQUEST to the UE, which shall respond with "
            "DEREGISTRATION ACCEPT unless the switch-off indicator makes an acknowledgement unnecessary.",
        ],
    )
    story.append(PageBreak())
    story += _clause(
        "10.2", "NAS timers",
        [
            "The following timers control retransmission and supervision of key NAS procedures at the UE "
            "side. Timer T3510 supervises the registration procedure; if it expires before REGISTRATION "
            "ACCEPT or REGISTRATION REJECT is received, the UE retransmits REGISTRATION REQUEST up to a "
            "configured number of times. Timer T3521 supervises the de-registration procedure in an analogous "
            "way.",
        ],
    )
    story.append(
        _make_table(
            ["Timer", "Default value", "Retransmission limit"],
            [
                ["T3510", "15 seconds", "5 attempts"],
                ["T3521", "15 seconds", "5 attempts"],
                ["T3511", "10 seconds", "N/A — cell re-selection timer"],
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story += _clause(
        "5.4.1", "Authentication procedure",
        [
            "The network may initiate NAS authentication at any time by sending an AUTHENTICATION REQUEST "
            "containing an authentication challenge. The UE calculates a response using its USIM and returns "
            "it in AUTHENTICATION RESPONSE. If the response does not match the expected value, the network "
            "sends AUTHENTICATION REJECT and the UE shall consider the USIM invalid for 5GS services until "
            "the UE is switched off or the USIM is removed.",
        ],
    )

    _build_pdf("ts_24501_nas_sample.pdf", "TS 24.501", "17.8.0", "17", "Non-Access-Stratum (NAS) protocol for 5G System (5GS)", story)


# ─────────────────────────────────────────────────────────────────────────
# 4. TS 38.321 — NR Medium Access Control (MAC) protocol
# ─────────────────────────────────────────────────────────────────────────

def build_ts_38321():
    story = []
    story += _clause(
        "5.1.2", "Random access procedure",
        [
            "The random access procedure may be contention-based or contention-free. In the four-step "
            "contention-based procedure, the UE transmits a random access preamble (Msg1), receives a random "
            "access response (Msg2) granting uplink resources, transmits a scheduled message containing UE "
            "identity (Msg3), and finally receives a contention resolution message (Msg4). The two-step "
            "procedure combines the preamble and Msg3 payload into a single MsgA transmission to reduce "
            "latency.",
        ],
    )
    story += _clause(
        "5.4.2", "HARQ operation",
        [
            "The MAC entity supports Hybrid Automatic Repeat reQuest (HARQ) with up to 16 parallel HARQ "
            "processes per serving cell for downlink and uplink respectively. Each HARQ process is identified "
            "by a HARQ process ID and maintains its own soft-combining buffer. Upon detecting a transport "
            "block error, the receiver shall request retransmission via HARQ feedback rather than waiting for "
            "RLC-layer retransmission, in order to minimise latency.",
        ],
    )
    story.append(PageBreak())
    story += _clause(
        "5.4.3", "Logical channel prioritization",
        [
            "When the MAC entity is granted uplink resources, it allocates them to logical channels according "
            "to a prioritized list, respecting each logical channel's Prioritised Bit Rate (PBR) and Bucket "
            "Size Duration (BSD) before serving remaining channels strictly by priority order.",
        ],
    )
    story += _clause(
        "Annex A", "Modulation and coding scheme (MCS) index table (informative excerpt)",
        [
            "The table below is an illustrative excerpt of the MCS index to modulation-order mapping used for "
            "PDSCH/PUSCH scheduling decisions referenced elsewhere in this specification.",
        ],
        level=2,
    )
    story.append(
        _make_table(
            ["MCS index", "Modulation order", "Modulation scheme"],
            [
                ["0", "2", "QPSK"],
                ["9", "4", "16QAM"],
                ["16", "6", "64QAM"],
                ["27", "8", "256QAM"],
            ],
        )
    )

    _build_pdf("ts_38321_mac_sample.pdf", "TS 38.321", "17.5.0", "17", "Medium Access Control (MAC) protocol specification", story)


# ─────────────────────────────────────────────────────────────────────────
# 5. TS 33.501 — Security architecture and procedures for 5G System
# ─────────────────────────────────────────────────────────────────────────

def build_ts_33501():
    story = []
    story += _clause(
        "6.1.3", "5G-AKA authentication procedure",
        [
            "5G-AKA is one of two authentication methods standardized for the 5G System, alongside EAP-AKA'. "
            "The AUSF requests authentication vectors from the UDM, which derives them using subscriber-"
            "specific keys stored in the ARPF. Upon successful completion, both the UE and the network derive "
            "a shared anchor key, K_SEAF, from which further session keys are derived. The SEAF forwards the "
            "authentication challenge to the UE via the AMF and verifies the UE's response before considering "
            "the subscriber authenticated.",
        ],
    )
    story += _clause(
        "6.2.1", "Key hierarchy",
        [
            "The 5G key hierarchy derives a chain of keys from the long-term subscriber key K. The "
            "Authentication Server Function derives K_AUSF, from which the Security Anchor Function derives "
            "K_SEAF. K_SEAF is in turn used by the AMF to derive K_AMF, from which NAS and Access Stratum "
            "keys (K_NASint, K_NASenc, K_gNB) are further derived for a given registration.",
        ],
    )
    story.append(PageBreak())
    story += _clause(
        "6.2.2", "Key derivation summary",
        [
            "The table below summarizes which entity derives each key in the hierarchy and from which parent "
            "key it is derived.",
        ],
    )
    story.append(
        _make_table(
            ["Key", "Derived by", "Derived from"],
            [
                ["K_AUSF", "AUSF / UE", "K (long-term subscriber key)"],
                ["K_SEAF", "SEAF / UE", "K_AUSF"],
                ["K_AMF", "AMF / UE", "K_SEAF"],
                ["K_gNB", "AMF / UE", "K_AMF"],
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story += _clause(
        "5.9.2", "User plane integrity protection",
        [
            "User plane integrity protection between the UE and the gNB is optional to support and to "
            "activate, and is subject to a maximum data rate above which integrity protection may be disabled "
            "for performance reasons, as configured by the network via the UE security capabilities exchange.",
        ],
    )

    _build_pdf("ts_33501_security_sample.pdf", "TS 33.501", "17.7.0", "17", "Security architecture and procedures for the 5G System", story)


def main():
    build_ts_38331()
    build_ts_23501()
    build_ts_24501()
    build_ts_38321()
    build_ts_33501()
    print(f"\nSample corpus generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
