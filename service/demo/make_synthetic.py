#!/usr/bin/env python3
"""Generate FICTIONAL discharge notes so the toolkit and the public demo can be exercised safely.

Every note produced here is synthesised from templates. No sentence, lab value, identifier or date
originates from a real record, and no real note was consulted to write the templates -- they encode
only the textbook presentation of each condition. That is the whole point: the demo on a public site
and the smoke test in CI must never touch clinical data, and a reader must be able to verify that by
reading this file.

Two outputs:
  --mode corpus    a few thousand notes with gold codes, enough for `icdassist fit` to run end to end
  --mode showcase  a handful of longer, hand-shaped cases for the website's demo

The notes omit the diagnosis statement, matching the `note_text_no_diagnosis` field the research used:
leaving the discharge diagnosis in the text leaks the label and inflates F1 roughly eightfold
(docs/withdiagnosis_results.md). A demo trained on the leaky field would look great and mean nothing.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---- presentations ---------------------------------------------------------------------------
# (key, codes, chief complaint pool, course pool). Codes are ICD-10-CM and are the ones a coder
# would plausibly assign to the templated text -- they are NOT drawn from any real coded record.
PRESENTATIONS = [
    ("pneumonia", ["J18.9", "R05.9"],
     ["productive cough and fever for {d} days", "fever with purulent sputum for {d} days",
      "cough, chills and pleuritic chest pain for {d} days"],
     ["Empirical ceftriaxone was started. Chest radiograph showed {lobe} consolidation. "
      "Fever subsided on hospital day {hd} and sputum production decreased.",
      "Treated with ampicillin-sulbactam and nebulised bronchodilators. Follow-up film showed "
      "partial resolution of the {lobe} infiltrate by day {hd}."]),
    ("uti_pyelo", ["N39.0", "N10"],
     ["dysuria and flank pain for {d} days", "fever with costovertebral angle tenderness for {d} days"],
     ["Urinalysis showed pyuria with nitrite positivity. Urine culture grew a gram-negative rod. "
      "Intravenous ceftriaxone was given and the patient became afebrile on day {hd}.",
      "Started on cefazolin after urine culture. Flank tenderness resolved by hospital day {hd}."]),
    ("chf_exac", ["I50.9", "R06.02", "I10"],
     ["progressive exertional dyspnoea and leg swelling for {d} days",
      "orthopnoea and weight gain of {kg} kg over {d} days"],
     ["Intravenous furosemide produced a net negative balance of {kg} kg. Jugular venous distension "
      "and bibasilar crackles improved. Oral diuretic was resumed before discharge on day {hd}.",
      "Diuresed with furosemide infusion; dyspnoea improved from NYHA class IV to class II by day {hd}."]),
    ("copd_exac", ["J44.1", "J44.9"],
     ["increased dyspnoea and sputum volume for {d} days in a long-standing smoker",
      "wheeze and breathlessness worse over {d} days"],
     ["Nebulised salbutamol and ipratropium with systemic corticosteroid were given. Peak flow "
      "improved and the patient was weaned off oxygen by hospital day {hd}.",
      "Treated with bronchodilators, a five-day prednisolone course and antibiotics; discharged on "
      "day {hd} on the prior inhaler regimen."]),
    ("dm_hyper", ["E11.65", "E11.9", "I10"],
     ["polyuria, polydipsia and fatigue for {d} days",
      "found to have a capillary glucose of {glu} mg/dL at a clinic visit"],
     ["Managed with an insulin sliding scale, then transitioned to basal-bolus. Glucose fell from "
      "{glu} to a fasting range of 130-160 mg/dL by hospital day {hd}. Dietitian review was arranged.",
      "Hydration and insulin were given; HbA1c returned at {a1c}%. Oral agents were adjusted before "
      "discharge on day {hd}."]),
    ("gi_bleed", ["K92.2", "K25.9", "D62"],
     ["melaena for {d} days with light-headedness", "coffee-ground vomiting and black stools for {d} days"],
     ["Haemoglobin was {hb} g/dL on arrival; {u} units of packed cells were transfused. Endoscopy "
      "showed a gastric ulcer with a clean base. Proton pump inhibitor infusion was given and no "
      "further bleeding occurred by day {hd}.",
      "Resuscitated and started on pantoprazole. Endoscopy demonstrated an ulcer without active "
      "bleeding; haemoglobin stabilised at {hb} g/dL by hospital day {hd}."]),
    ("stroke_isch", ["I63.9", "I10", "E78.5"],
     ["sudden right-sided weakness and slurred speech {d} hours before arrival",
      "acute onset left hemiparesis noted on waking"],
     ["Imaging showed an acute infarct without haemorrhage. Antiplatelet therapy and a statin were "
      "started, and rehabilitation began on hospital day {hd}. Power improved from {mrc}/5 to 4/5.",
      "Admitted to the stroke unit. Swallow screening was passed on day 2 and physiotherapy "
      "commenced; by day {hd} the patient ambulated with a frame."]),
    ("cellulitis", ["L03.115", "E11.9"],
     ["a painful red swollen left leg for {d} days", "spreading erythema of the right lower leg for {d} days"],
     ["Limb elevation and intravenous cefazolin were given; erythema margins regressed by hospital "
      "day {hd} and the patient was switched to oral therapy.",
      "Treated with intravenous antibiotics; no abscess was seen on bedside ultrasound. Discharged "
      "on day {hd} on oral cephalexin."]),
    ("ckd_aki", ["N17.9", "N18.4", "E87.5"],
     ["reduced urine output and nausea for {d} days", "malaise with rising creatinine noted at dialysis clinic"],
     ["Creatinine peaked at {cr} mg/dL with potassium {k} mmol/L. Nephrotoxic agents were stopped and "
      "volume was repleted; creatinine fell to baseline by hospital day {hd}.",
      "Managed with fluid optimisation and potassium binders; urine output recovered by day {hd}."]),
    ("cholecystitis", ["K81.0", "K80.20"],
     ["right upper quadrant pain and vomiting for {d} days", "epigastric pain radiating to the back for {d} days"],
     ["Ultrasound showed gallstones with a thickened gallbladder wall. Laparoscopic cholecystectomy "
      "was performed on hospital day {hd} without complication.",
      "Treated conservatively with antibiotics and analgesia; pain settled and interval surgery was "
      "planned at discharge on day {hd}."]),
    ("afib", ["I48.91", "I50.9", "Z79.01"],
     ["palpitations and breathlessness for {d} days", "an irregular pulse found during a routine review"],
     ["Rate control with a beta-blocker was achieved; ventricular rate fell from {hr} to 80 bpm. "
      "Anticoagulation was initiated after stroke-risk assessment on day {hd}.",
      "Rhythm remained atrial fibrillation with controlled rate. Oral anticoagulant was started and "
      "the patient was discharged on hospital day {hd}."]),
    ("anemia_iron", ["D50.9", "K92.1"],
     ["progressive tiredness and exertional dyspnoea over {d} weeks",
      "pallor and intermittent dark stools for {d} weeks"],
     ["Haemoglobin was {hb} g/dL with a ferritin of {fer} ng/mL. Intravenous iron was given and "
      "colonoscopy was arranged as an outpatient. Discharged on day {hd}.",
      "Transfused {u} units and started on iron replacement; symptoms improved by hospital day {hd}."]),
    ("sepsis_uro", ["A41.9", "N39.0", "R65.21"],
     ["fever, rigors and confusion for {d} days", "collapse at home with fever and tachycardia"],
     ["Lactate was {lac} mmol/L with a blood pressure of {sbp}/60 mmHg. Fluid resuscitation and "
      "broad-spectrum antibiotics were given; vasopressor support was weaned by hospital day {hd}.",
      "Treated as urosepsis with intravenous antibiotics and fluids; blood cultures grew a "
      "gram-negative organism. Haemodynamics normalised by day {hd}."]),
    ("fracture_hip", ["S72.001A", "W19.XXXA", "M81.0"],
     ["a fall at home with inability to bear weight on the left hip",
      "a mechanical fall with immediate left groin pain"],
     ["Radiographs showed a femoral neck fracture. Hemiarthroplasty was performed on hospital day "
      "{hd} and mobilisation with a frame began the next day.",
      "Underwent internal fixation on day {hd}; postoperative physiotherapy progressed to a "
      "walking frame before transfer to rehabilitation."]),
]

COMORBID = [  # extra codes added with some probability, with the sentence that justifies them
    (["I10"], "Hypertension has been treated with amlodipine for {y} years."),
    (["E11.9"], "Type 2 diabetes mellitus has been managed with metformin for {y} years."),
    (["E78.5"], "Dyslipidaemia is treated with a statin."),
    (["Z72.0"], "The patient has a {py} pack-year smoking history."),
    (["N18.3"], "Chronic kidney disease stage 3 is followed in the renal clinic."),
    (["I25.10"], "Coronary artery disease was documented on a previous angiogram."),
    (["F32.9"], "A depressive episode was treated two years ago."),
    (["M17.11"], "Right knee osteoarthritis limits walking distance."),
]

LOBES = ["right lower lobe", "left lower lobe", "right middle lobe", "bilateral basal"]

# Vitals and labs must not contradict the narrative. A fabricated note that reports haemoglobin
# 13.7 g/dL and then says it "stabilised at 6.8" is the kind of detail a clinician reading the demo
# notices immediately, and it would make the whole page look careless. So each presentation pins the
# few values its story depends on; everything else stays random within a normal-ish range.
RANGES = {
    "pneumonia":     {"temp": (38.2, 39.6), "wbc": (12.0, 19.8), "crp": (60.0, 220.0), "spo2": (88, 94)},
    "uti_pyelo":     {"temp": (38.0, 39.4), "wbc": (11.0, 18.0), "crp": (40.0, 180.0)},
    "chf_exac":      {"spo2": (88, 94), "rr": (22, 30), "temp": (36.2, 37.2)},
    "copd_exac":     {"spo2": (86, 93), "rr": (22, 32), "temp": (36.4, 37.8)},
    "dm_hyper":      {"na": (128, 136), "temp": (36.2, 37.4)},
    "gi_bleed":      {"hb": (5.8, 8.6), "sbp": (88, 112), "pulse": (96, 130), "temp": (36.1, 37.2)},
    "stroke_isch":   {"temp": (36.2, 37.3), "sbp": (150, 196)},
    "cellulitis":    {"temp": (37.6, 39.0), "wbc": (10.5, 17.0), "crp": (40.0, 160.0)},
    "ckd_aki":       {"cr": (2.4, 6.8), "k": (5.1, 6.4), "temp": (36.2, 37.3)},
    "cholecystitis": {"temp": (37.8, 39.2), "wbc": (11.0, 18.5), "crp": (50.0, 190.0)},
    "afib":          {"pulse": (118, 168), "temp": (36.2, 37.2)},
    "anemia_iron":   {"hb": (6.2, 8.8), "pulse": (92, 118), "temp": (36.2, 37.1)},
    "sepsis_uro":    {"temp": (38.6, 40.1), "wbc": (14.0, 24.0), "crp": (120.0, 320.0),
                      "sbp": (78, 98), "pulse": (108, 140), "rr": (22, 32)},
    "fracture_hip":  {"temp": (36.2, 37.2), "hb": (9.0, 12.0)},
}
NORMAL = {"temp": (36.2, 38.4), "pulse": (62, 112), "sbp": (102, 168), "dbp": (54, 96),
          "rr": (14, 24), "spo2": (94, 99), "wbc": (4.0, 13.0), "hb": (10.4, 15.1),
          "plt": (98, 420), "cr": (0.6, 1.6), "na": (134, 145), "k": (3.4, 5.0),
          "crp": (0.4, 40.0)}


def _v(key, which, rng, ndigits=1):
    lo, hi = RANGES.get(key, {}).get(which, NORMAL[which])
    v = rng.uniform(lo, hi)
    return round(v, ndigits) if ndigits else int(round(v))


def _fill(tpl, rng):
    return tpl.format(
        d=rng.randint(2, 10), hd=rng.randint(3, 9), kg=rng.randint(2, 7),
        glu=rng.randint(280, 520), a1c=round(rng.uniform(8.5, 13.0), 1),
        hb=round(rng.uniform(5.8, 9.4), 1), u=rng.randint(1, 4),
        cr=round(rng.uniform(2.4, 6.8), 1), k=round(rng.uniform(5.1, 6.4), 1),
        hr=rng.randint(120, 170), fer=rng.randint(3, 18), lac=round(rng.uniform(2.2, 5.6), 1),
        sbp=rng.randint(78, 98), mrc=rng.randint(1, 3), y=rng.randint(3, 20),
        py=rng.randint(20, 60), lobe=rng.choice(LOBES))


def make_note(rng):
    key, codes, cc_pool, course_pool = rng.choice(PRESENTATIONS)
    codes = list(codes)
    age, sex = rng.randint(41, 89), rng.choice(["man", "woman"])
    cc = _fill(rng.choice(cc_pool), rng)
    course = _fill(rng.choice(course_pool), rng)

    past = []
    for extra_codes, sent in COMORBID:
        if rng.random() < 0.28:
            past.append(_fill(sent, rng))
            for c in extra_codes:
                if c not in codes:
                    codes.append(c)
    if not past:
        past.append("No chronic illness is documented.")

    spo2 = _v(key, "spo2", rng, 0)
    sbp = _v(key, "sbp", rng, 0)
    dbp = max(40, sbp - rng.randint(28, 58))     # derived, not independent: 89/83 mmHg is not a thing
    vitals = (f"Temperature {_v(key, 'temp', rng)} C, pulse {_v(key, 'pulse', rng, 0)}/min, "
              f"blood pressure {sbp}/{dbp} mmHg, "
              f"respiratory rate {_v(key, 'rr', rng, 0)}/min, SpO2 {spo2}% on "
              f"{'room air' if spo2 >= 94 else '2 L/min nasal cannula'}.")
    labs = (f"WBC {_v(key, 'wbc', rng)} x10^3/uL, haemoglobin {_v(key, 'hb', rng)} g/dL, "
            f"platelets {_v(key, 'plt', rng, 0)} x10^3/uL, creatinine {_v(key, 'cr', rng, 2)} mg/dL, "
            f"sodium {_v(key, 'na', rng, 0)} mmol/L, potassium {_v(key, 'k', rng)} mmol/L, "
            f"CRP {_v(key, 'crp', rng)} mg/L.")

    art = "An" if str(age)[0] in "8" else "A"      # "An 81-year-old", not "A 81-year-old"
    text = "\n".join([
        f"CHIEF COMPLAINT: {art} {age}-year-old {sex} presented with {cc}.",
        f"PAST HISTORY: {' '.join(past)}",
        f"PHYSICAL EXAMINATION: {vitals}",
        f"LABORATORY: {labs}",
        f"HOSPITAL COURSE: {course}",
        "CONDITION AT DISCHARGE: Stable, ambulating, tolerating an oral diet. Outpatient follow-up "
        "was arranged and discharge medications were reconciled with the patient.",
    ])
    return key, text, sorted(set(codes))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["corpus", "showcase"], default="corpus")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = 12 if args.mode == "showcase" else args.n
    seen = set()
    rows = []
    while len(rows) < n:
        key, text, codes = make_note(rng)
        if args.mode == "showcase" and key in seen:
            continue                      # one of each presentation, so the demo spans the book
        seen.add(key)
        rows.append({"id": f"SYN-{len(rows):05d}", "presentation": key, "text": text, "codes": codes,
                     "synthetic": True})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    codes = {c for r in rows for c in r["codes"]}
    print(f"wrote {len(rows)} FICTIONAL notes, {len(codes)} distinct codes -> {out}")
    print(f"  median length {sorted(len(r['text']) for r in rows)[len(rows) // 2]} chars")


if __name__ == "__main__":
    main()
