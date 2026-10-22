"""Published thesis results (Tabel 4.1-4.3, DRAFT SKRIPSI COMPLETED), used as parity targets.

Attribute names were reconstructed from the scale factors where the PDF lost the symbols; the
parity run confirms or corrects them (vault 05-Literature/thesis-summary.md).
"""

# Winning pair of each thesis row, as legacy column names, and the transform set that produced it.
# Confirmed by exact reproduction (r, theta, SF) on 2026-10-20. SWT-LBR used the later notebook
# version (sqrt transform); the other five used the thesis-era set (x^-2).
THESIS_PAIRS = {
    ("PHIE", "UBR"): ("VP/VS^2", "ln(LAMDA_MIN_MU)", "thesis"),
    ("PHIE", "LBR"): ("ln(IP)", "IS^-2", "thesis"),
    ("VSH", "UBR"): ("1/LAMDA_RHO", "POISSON_RATIO^2", "thesis"),
    ("VSH", "LBR"): ("1/IP", "IS^-2", "thesis"),
    ("SWT", "UBR"): ("MU_RHO", "LAMDA_PER_MU^2", "thesis"),
    ("SWT", "LBR"): ("sqrt(IP)", "1/MU_RHO", "final"),
}

THESIS_TABLES = {
    ("PHIE", "UBR"): dict(sf1=2.726666075, sf2=0.742202817, theta=-25, m=0.107709206,
                          c=-0.749707843, r=0.415083517),
    ("PHIE", "LBR"): dict(sf1=1.382488257, sf2=1088687492.0, theta=-97, m=0.352072485,
                          c=5.276223026, r=0.6841823),
    ("VSH", "UBR"): dict(sf1=1714604708.75823, sf2=99.16826743, theta=49, m=0.132058697,
                         c=-1.147806382, r=0.780790287),
    ("VSH", "LBR"): dict(sf1=222066.0352, sf2=1088687492.0, theta=-34, m=0.165051448,
                         c=0.352113652, r=0.901447974),
    ("SWT", "UBR"): dict(sf1=5.46519188293463e-08, sf2=1.223634021, theta=-98, m=0.044327439,
                         c=1.288983146, r=0.628761034),
    ("SWT", "LBR"): dict(sf1=0.066075935, sf2=1088687492.0, theta=61, m=0.23507974,
                         c=-2.233140111, r=0.756504634),
}
