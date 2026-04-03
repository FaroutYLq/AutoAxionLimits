"""
Pipeline configuration: coupling types, arXiv keywords, and physical corrections.
"""

# Maps each coupling type to its repository artifacts.
COUPLING_TYPES = {
    "DarkPhoton": {
        "class_name": "DarkPhoton",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/DarkPhoton",
        "notebooks": ["DarkPhoton.ipynb"],
        "docs_file": "docs/dp.md",
        "axes": {"x": "mass [eV]", "y": "kinetic mixing chi"},
    },
    "AxionPhoton": {
        "class_name": "AxionPhoton",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionPhoton",
        "notebooks": [
            "AxionPhoton.ipynb",
            "AxionPhoton_Ultralight.ipynb",
            "AxionPhoton_ColliderBounds.ipynb",
            "AxionPhoton_Closeups.ipynb",
        ],
        "docs_file": "docs/ap.md",
        "axes": {"x": "mass [eV]", "y": "g_agamma [GeV^-1]"},
    },
    "AxionElectron": {
        "class_name": "AxionElectron",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionElectron",
        "notebooks": ["AxionElectron.ipynb"],
        "docs_file": "docs/ae.md",
        "axes": {"x": "mass [eV]", "y": "g_ae"},
    },
    "AxionNeutron": {
        "class_name": "AxionNeutron",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionNeutron",
        "notebooks": ["AxionNeutron.ipynb"],
        "docs_file": "docs/an.md",
        "axes": {"x": "mass [eV]", "y": "g_an"},
    },
    "AxionProton": {
        "class_name": "AxionProton",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionProton",
        "notebooks": ["AxionProton.ipynb"],
        "docs_file": "docs/aprot.md",
        "axes": {"x": "mass [eV]", "y": "g_ap"},
    },
    "AxionEDM": {
        "class_name": "AxionEDM",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionEDM",
        "notebooks": ["AxionEDM.ipynb"],
        "docs_file": "docs/aedm.md",
        "axes": {"x": "mass [eV]", "y": "d_n [e cm]"},
    },
    "AxionCPV": {
        "class_name": "AxionCPV",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionCPV",
        "notebooks": ["AxionCPV.ipynb"],
        "docs_file": "docs/acpv.md",
        "axes": {"x": "mass [eV]", "y": "coupling"},
    },
    "AxionMass": {
        "class_name": "AxionMass",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/AxionMass",
        "notebooks": ["AxionMass.ipynb"],
        "docs_file": "docs/am.md",
        "axes": {"x": "f_a [GeV]", "y": "m_a [eV]"},
    },
    "MonopoleDipole": {
        "class_name": "MonopoleDipole",
        "plotfuncs_file": "PlotFuncs.py",
        "data_dir": "limit_data/MonopoleDipole",
        "notebooks": ["AxionNeutron.ipynb"],
        "docs_file": "docs/md.md",
        "axes": {"x": "mass [eV]", "y": "coupling"},
    },
    "ScalarPhoton": {
        "class_name": "ScalarPhoton",
        "plotfuncs_file": "PlotFuncs_ScalarVector.py",
        "data_dir": "limit_data/ScalarPhoton",
        "notebooks": ["Scalars.ipynb"],
        "docs_file": "docs/scalar.md",
        "axes": {"x": "mass [eV]", "y": "d_e"},
    },
    "ScalarElectron": {
        "class_name": "ScalarElectron",
        "plotfuncs_file": "PlotFuncs_ScalarVector.py",
        "data_dir": "limit_data/ScalarElectron",
        "notebooks": ["Scalars.ipynb"],
        "docs_file": "docs/scalar.md",
        "axes": {"x": "mass [eV]", "y": "d_e"},
    },
    "ScalarBaryon": {
        "class_name": "ScalarBaryon",
        "plotfuncs_file": "PlotFuncs_ScalarVector.py",
        "data_dir": "limit_data/ScalarBaryon",
        "notebooks": ["Scalars.ipynb"],
        "docs_file": "docs/scalar.md",
        "axes": {"x": "mass [eV]", "y": "coupling"},
    },
    "ScalarNucleon": {
        "class_name": "ScalarNucleon",
        "plotfuncs_file": "PlotFuncs_ScalarVector.py",
        "data_dir": "limit_data/ScalarNucleon",
        "notebooks": ["Scalars.ipynb"],
        "docs_file": "docs/scalar.md",
        "axes": {"x": "mass [eV]", "y": "coupling"},
    },
    "VectorBL": {
        "class_name": "VectorBL",
        "plotfuncs_file": "PlotFuncs_ScalarVector.py",
        "data_dir": "limit_data/VectorB-L",
        "notebooks": ["Vectors.ipynb"],
        "docs_file": "docs/vector.md",
        "axes": {"x": "mass [eV]", "y": "g_BL"},
    },
}

# Per-coupling arXiv keyword filters (pre-Claude screening).
ARXIV_KEYWORDS = {
    "DarkPhoton": [
        "dark photon",
        "kinetic mixing",
        "hidden photon",
        "dark photon search",
        "dark photon limit",
        "paraphoton",
        "U(1) dark photon",
    ],
    "AxionPhoton": [
        "axion photon",
        "ALP photon",
        "photon coupling",
        "ADMX",
        "CAST",
        "HAYSTAC",
        "haloscope",
        "axion dark matter",
        "gagg",
        "g_agamma",
        "axion-like particle photon",
    ],
    "AxionElectron": [
        "axion electron",
        "ALP electron",
        "gaee",
        "solar axion",
        "axion solar",
        "electron coupling axion",
    ],
    "AxionNeutron": [
        "axion neutron",
        "ALP neutron",
        "gann",
        "neutron EDM axion",
        "axion nucleon",
        "spin-dependent axion",
    ],
    "AxionProton": [
        "axion proton",
        "ALP proton",
        "gapp",
        "axion nucleon",
        "proton coupling axion",
    ],
    "AxionEDM": [
        "axion EDM",
        "electric dipole moment axion",
        "axion CP violation",
        "oscillating EDM",
    ],
    "AxionCPV": [
        "axion CP violation",
        "axion CPV",
        "CP-violating axion",
        "axion theta",
        "strong CP",
    ],
    "AxionMass": [
        "axion mass",
        "axion decay constant",
        "f_a axion",
        "axion mass bound",
        "axion mass limit",
    ],
    "MonopoleDipole": [
        "monopole dipole",
        "fifth force axion",
        "spin-mass coupling",
        "monopole-dipole",
        "g_p g_s",
    ],
    "ScalarPhoton": [
        "scalar photon",
        "dilaton photon",
        "scalar field photon coupling",
        "moduli photon",
    ],
    "ScalarElectron": [
        "scalar electron",
        "dilaton electron",
        "scalar dark matter electron",
        "d_e scalar",
    ],
    "ScalarBaryon": [
        "scalar baryon",
        "scalar dark matter baryon",
        "baryon coupling scalar",
    ],
    "ScalarNucleon": [
        "scalar nucleon",
        "dilaton nucleon",
        "scalar dark matter nucleon",
    ],
    "VectorBL": [
        "B-L gauge boson",
        "vector B-L",
        "baryon minus lepton",
        "U(1)_{B-L}",
        "B minus L boson",
        "dark vector B-L",
    ],
}

# Physical corrections passed verbatim to the reviewer agent.
PHYSICAL_CORRECTIONS = {
    "DarkPhoton": {
        "dm_density": {
            "repo_convention": 0.45,  # GeV/cm^3
            "common_paper_values": [0.3, 0.4, 0.45],
            "formula": "chi_corrected = chi_paper * sqrt(rho_repo / rho_paper)",
            "description": (
                "Many dark photon haloscope limits assume rho_DM = 0.3 or 0.4 GeV/cm^3. "
                "This repo uses 0.45 GeV/cm^3. Apply sqrt(rho_repo/rho_paper) factor."
            ),
        },
        "polarization": {
            "description": (
                "Some haloscopes assume a fixed single polarisation direction. "
                "If the paper assumes a specific polarisation fraction, "
                "flag this for human review — the correction factor depends on geometry."
            ),
        },
        "local_velocity": {
            "description": (
                "A few experiments use v_0 = 220 km/s or different velocity distributions. "
                "Flag if the paper's assumed halo model differs significantly from standard SHM."
            ),
        },
    },
    "AxionPhoton": {
        "dm_density": {
            "repo_convention": 0.45,
            "common_paper_values": [0.3, 0.4, 0.45],
            "formula": "g_corrected = g_paper * sqrt(rho_repo / rho_paper)",
            "description": (
                "Axion haloscope limits scale as sqrt(rho_DM). "
                "Apply sqrt(rho_repo/rho_paper) when paper uses a different density."
            ),
        },
    },
    "AxionElectron": {
        "dm_density": {
            "repo_convention": 0.45,
            "formula": "g_corrected = g_paper * sqrt(rho_repo / rho_paper)",
            "description": "Same sqrt(rho_DM) scaling for axion-electron DM search limits.",
        },
    },
    "AxionNeutron": {
        "dm_density": {
            "repo_convention": 0.45,
            "formula": "g_corrected = g_paper * sqrt(rho_repo / rho_paper)",
            "description": "Same sqrt(rho_DM) scaling for axion-neutron DM search limits.",
        },
    },
    "AxionProton": {
        "dm_density": {
            "repo_convention": 0.45,
            "formula": "g_corrected = g_paper * sqrt(rho_repo / rho_paper)",
            "description": "Same sqrt(rho_DM) scaling for axion-proton DM search limits.",
        },
    },
}

# arXiv categories to search in.
ARXIV_CATEGORIES = ["hep-ph", "hep-ex", "astro-ph.CO", "astro-ph.HE", "physics.ins-det"]

# Minimum extraction confidence to open a PR (lower → tagged [LOW CONFIDENCE]).
LOW_CONFIDENCE_THRESHOLD = 0.6

# Number of papers to process per daily run.
MAX_PAPERS_PER_RUN = 5

# ---------------------------------------------------------------------------
# Backfill configuration
# ---------------------------------------------------------------------------
BACKFILL_MAX_PAPERS_PER_RUN = 10
BACKFILL_DEFAULT_MIN_CITATIONS = 10

# Curated search queries for INSPIRE-HEP (per coupling type).
# These are used with INSPIRE's SPIRES syntax: find (t "..." or abs "...").
# Keep queries short (2-3 key terms) for good recall; precision comes from
# the subsequent keyword classification and LLM filters.
INSPIRE_SEARCH_QUERIES = {
    "DarkPhoton": [
        "dark photon",
        "hidden photon",
        "kinetic mixing",
    ],
    "AxionPhoton": [
        "axion photon coupling",
        "haloscope",
        "axion-like particle",
    ],
    "AxionElectron": [
        "axion electron",
        "solar axion",
    ],
    "AxionNeutron": [
        "axion neutron",
        "axion nucleon",
    ],
    "AxionProton": [
        "axion proton",
    ],
    "AxionEDM": [
        "axion electric dipole moment",
        "oscillating EDM",
    ],
    "AxionCPV": [
        "axion CP violation",
        "strong CP",
    ],
    "AxionMass": [
        "axion mass",
        "axion decay constant",
    ],
    "MonopoleDipole": [
        "monopole-dipole",
        "spin-mass coupling",
    ],
    "ScalarPhoton": [
        "scalar photon coupling",
        "dilaton photon",
    ],
    "ScalarElectron": [
        "scalar electron coupling",
        "dilaton electron",
    ],
    "ScalarBaryon": [
        "scalar baryon",
    ],
    "ScalarNucleon": [
        "scalar nucleon",
        "dilaton nucleon",
    ],
    "VectorBL": [
        "B-L gauge boson",
        "B-L boson",
    ],
}
