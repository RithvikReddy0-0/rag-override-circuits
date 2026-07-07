"""Minimal-pair construction for context-vs-parametric conflict experiments.

Each Pair is one legal question with two versions of retrieved context:
  - context_supportive: states a fact (`parametric_answer`) hypothesized to match
    the model's parametric prior.
  - context_conflicting: the SAME passage with a single controlled edit (a
    number swapped, or a clause negated) so the context-grounded answer
    becomes `edited_answer` instead.

`has_strong_prior=False` marks the two section-number pairs: there is no
real-world reason a model would have a parametric belief about an arbitrary
contract's internal section numbering, so those pairs are copy-fidelity
controls, not genuine parametric-vs-context conflict tests. Everything else
is chosen to have a plausible pretraining-era prior (common statutory
defaults, legal maxims), so a real conflict can exist per CLAUDE.md.

Every `parametric_answer` / `edited_answer` must be exactly one token in both
GPT-2-small's and Pythia-160M's vocabularies (checked by `validate_pairs`),
because the pilot's behavioral metric is a logit difference between two
single tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

EditType = Literal["numeric_swap", "negation", "section_number"]

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = REPO_ROOT / "data" / "pairs" / "pairs.jsonl"


@dataclass(frozen=True)
class Pair:
    pair_id: str
    category: str
    edit_type: EditType
    has_strong_prior: bool
    question: str
    context_supportive: str
    context_conflicting: str
    parametric_answer: str
    edited_answer: str
    notes: str = ""


# Hand-written pairs. Wording was chosen, then checked empirically against
# both tokenizers (see validate_pairs) — every answer below is confirmed to
# be a single token in both GPT-2-small and Pythia-160M.
PAIRS: list[Pair] = [
    Pair(
        pair_id="notice_period",
        category="contract_notice",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, how many days of notice must be given before terminating the agreement?",
        context_supportive=(
            "This Service Agreement may be terminated by either party upon written notice. "
            "Under Section 4.2, the required notice period before termination is 30 days."
        ),
        context_conflicting=(
            "This Service Agreement may be terminated by either party upon written notice. "
            "Under Section 4.2, the required notice period before termination is 7 days."
        ),
        parametric_answer="30",
        edited_answer="7",
    ),
    Pair(
        pair_id="age_of_majority",
        category="civil_status",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, at what age does a person reach the age of majority?",
        context_supportive=(
            "Under the Family Code, a person reaches the age of majority, and is considered "
            "a legal adult for all purposes, at 18 years of age."
        ),
        context_conflicting=(
            "Under the Family Code, a person reaches the age of majority, and is considered "
            "a legal adult for all purposes, at 21 years of age."
        ),
        parametric_answer="18",
        edited_answer="21",
    ),
    Pair(
        pair_id="drinking_age",
        category="civil_status",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, what is the minimum legal age to purchase alcohol?",
        context_supportive=(
            "Under the state Alcoholic Beverage Control Act, the minimum legal age to "
            "purchase or consume alcohol is 21."
        ),
        context_conflicting=(
            "Under the state Alcoholic Beverage Control Act, the minimum legal age to "
            "purchase or consume alcohol is 18."
        ),
        parametric_answer="21",
        edited_answer="18",
    ),
    Pair(
        pair_id="copyright_term",
        category="ip_law",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, copyright protection lasts for the life of the author plus how many years?",
        context_supportive=(
            "Under the Copyright Act, copyright protection for a work created by an "
            "individual author lasts for the life of the author plus 70 years."
        ),
        context_conflicting=(
            "Under the Copyright Act, copyright protection for a work created by an "
            "individual author lasts for the life of the author plus 50 years."
        ),
        parametric_answer="70",
        edited_answer="50",
    ),
    Pair(
        pair_id="jury_size",
        category="criminal_procedure",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, how many jurors must a felony criminal jury consist of?",
        context_supportive=(
            "Under the Criminal Procedure Code, a jury empaneled to try a felony criminal "
            "case shall consist of 12 jurors."
        ),
        context_conflicting=(
            "Under the Criminal Procedure Code, a jury empaneled to try a felony criminal "
            "case shall consist of 6 jurors."
        ),
        parametric_answer="12",
        edited_answer="6",
    ),
    Pair(
        pair_id="voting_age",
        category="civil_status",
        edit_type="numeric_swap",
        has_strong_prior=True,
        question="According to the passage, at what age does a citizen become eligible to vote?",
        context_supportive=(
            "Under the Electoral Act, a citizen is eligible to vote in national elections "
            "upon reaching 18 years of age."
        ),
        context_conflicting=(
            "Under the Electoral Act, a citizen is eligible to vote in national elections "
            "upon reaching 21 years of age."
        ),
        parametric_answer="18",
        edited_answer="21",
    ),
    Pair(
        pair_id="probation_period",
        category="employment",
        edit_type="numeric_swap",
        has_strong_prior=False,
        notes="Weaker/uncertain prior than the civil-status facts above; prior-check will confirm.",
        question="According to the passage, how many days is the employee's initial probationary period?",
        context_supportive=(
            "Under the Employment Agreement, a newly hired employee is subject to an "
            "initial probationary period of 90 days."
        ),
        context_conflicting=(
            "Under the Employment Agreement, a newly hired employee is subject to an "
            "initial probationary period of 30 days."
        ),
        parametric_answer="90",
        edited_answer="30",
    ),
    Pair(
        pair_id="noncompete_duration",
        category="employment",
        edit_type="numeric_swap",
        has_strong_prior=False,
        notes="Weaker/uncertain prior than the civil-status facts above; prior-check will confirm.",
        question="According to the passage, for how many years does the non-compete restriction remain in effect?",
        context_supportive=(
            "Under the Non-Compete Agreement, the restriction on the employee working for "
            "a competitor remains in effect for 1 year after termination."
        ),
        context_conflicting=(
            "Under the Non-Compete Agreement, the restriction on the employee working for "
            "a competitor remains in effect for 5 years after termination."
        ),
        parametric_answer="1",
        edited_answer="5",
    ),
    Pair(
        pair_id="presumption_of_innocence",
        category="criminal_procedure",
        edit_type="negation",
        has_strong_prior=True,
        notes="Question rephrased as Yes/No (rather than 'innocent or guilty?') to avoid a word-order/recency confound where the model favors whichever option is named last in the question regardless of context.",
        question="According to the passage, is a person charged with an offense presumed innocent?",
        context_supportive=(
            "Under the Criminal Code, every person charged with an offense is presumed "
            "innocent until proven guilty according to law."
        ),
        context_conflicting=(
            "Under the Criminal Code, every person charged with an offense is presumed "
            "guilty until proven innocent according to law."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="right_to_attorney",
        category="criminal_procedure",
        edit_type="negation",
        has_strong_prior=True,
        question="According to the passage, does a person under arrest have the right to consult an attorney before questioning?",
        context_supportive=(
            "Under the Suspects' Rights Act, a person under arrest has the right to "
            "consult an attorney before being questioned by police."
        ),
        context_conflicting=(
            "Under the Suspects' Rights Act, a person under arrest does not have the "
            "right to consult an attorney before being questioned by police."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="right_to_remain_silent",
        category="criminal_procedure",
        edit_type="negation",
        has_strong_prior=True,
        question="According to the passage, does a person under arrest have the right to remain silent?",
        context_supportive=(
            "Under the Suspects' Rights Act, a person under arrest has the right to "
            "remain silent and is not required to answer questions."
        ),
        context_conflicting=(
            "Under the Suspects' Rights Act, a person under arrest does not have the "
            "right to remain silent and is required to answer questions."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="settlement_confidentiality",
        category="civil_procedure",
        edit_type="negation",
        has_strong_prior=False,
        notes="Weaker/uncertain prior than the rights-based negations above; prior-check will confirm. Question rephrased as Yes/No (rather than 'confidential or public?') to avoid a word-order/recency confound.",
        question="According to the passage, are the terms of the settlement confidential?",
        context_supportive=(
            "Under the Settlement Agreement, the terms and amount of the settlement are "
            "confidential and may not be disclosed to third parties."
        ),
        context_conflicting=(
            "Under the Settlement Agreement, the terms and amount of the settlement are "
            "public and may be disclosed to third parties."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="at_will_employment",
        category="employment",
        edit_type="negation",
        has_strong_prior=False,
        notes="Weaker/uncertain prior than the rights-based negations above; prior-check will confirm.",
        question="According to the passage, may the employer terminate the employee without cause?",
        context_supportive=(
            "Under the Employment Agreement, the employment is at-will, and the employer "
            "may terminate the employee at any time without cause."
        ),
        context_conflicting=(
            "Under the Employment Agreement, the employment is not at-will, and the "
            "employer may only terminate the employee for cause."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="subletting_permission",
        category="lease",
        edit_type="negation",
        has_strong_prior=False,
        notes="Weaker/uncertain prior than the rights-based negations above; prior-check will confirm. Question rephrased as Yes/No (rather than 'permitted or prohibited?') to avoid a word-order/recency confound.",
        question="According to the passage, is the tenant permitted to sublet the premises?",
        context_supportive=(
            "Under the Lease Agreement, the tenant is permitted to sublet the premises "
            "with the landlord's prior written consent."
        ),
        context_conflicting=(
            "Under the Lease Agreement, the tenant is prohibited from subletting the "
            "premises under any circumstances."
        ),
        parametric_answer="Yes",
        edited_answer="No",
    ),
    Pair(
        pair_id="indemnification_section",
        category="section_reference",
        edit_type="section_number",
        has_strong_prior=False,
        notes="No real parametric prior over an arbitrary section number; this is a copy-fidelity control, not a conflict test.",
        question="According to the passage, in which section are the indemnification obligations set forth?",
        context_supportive=(
            "This Agreement contains an indemnification clause. The indemnification "
            "obligations of the parties are set forth in Section 8 of this Agreement."
        ),
        context_conflicting=(
            "This Agreement contains an indemnification clause. The indemnification "
            "obligations of the parties are set forth in Section 3 of this Agreement."
        ),
        parametric_answer="8",
        edited_answer="3",
    ),
    Pair(
        pair_id="governing_law_section",
        category="section_reference",
        edit_type="section_number",
        has_strong_prior=False,
        notes="No real parametric prior over an arbitrary section number; this is a copy-fidelity control, not a conflict test.",
        question="According to the passage, in which section is the governing law provision specified?",
        context_supportive=(
            "This Agreement is governed by the laws of the state in which it was "
            "executed. The governing law provision is specified in Section 12 of this Agreement."
        ),
        context_conflicting=(
            "This Agreement is governed by the laws of the state in which it was "
            "executed. The governing law provision is specified in Section 2 of this Agreement."
        ),
        parametric_answer="12",
        edited_answer="2",
    ),
]


# =========================================================================
# Phase 1: templated numeric_swap candidates.
#
# Direction is NOT assumed: for every candidate we probe the model's bare
# prior between value_a and value_b (src/priors.py); the measured winner
# becomes answer_prior_token and the loser becomes answer_context_token
# (what the conflicting context states). Candidates are filtered per model:
#   1. value_a and value_b must each be a single token in the model's vocab;
#   2. supportive/conflicting prompts must tokenize to the same length and
#      differ at exactly ONE token position (the critical number);
#   3. |bare-prior logit diff| >= priors.PRIOR_THRESHOLD.
# The kept set may therefore differ between GPT-2 and Pythia — both are
# written out and both attritions logged.
# =========================================================================


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    domain: str
    question: str
    template: str  # context passage with a single "{value}" slot
    value_a: str
    value_b: str

    def context_for(self, value: str) -> str:
        return self.template.format(value=value)


PHASE1_CANDIDATES: list[Candidate] = [
    # ---- ages -----------------------------------------------------------
    Candidate("p1_voting_age", "ages",
              "According to the passage, at what age does a citizen become eligible to vote?",
              "Under the Electoral Act, a citizen is eligible to vote in national elections upon reaching {value} years of age.",
              "18", "21"),
    Candidate("p1_age_of_majority", "ages",
              "According to the passage, at what age does a person reach the age of majority?",
              "Under the Family Code, a person reaches the age of majority and is considered a legal adult for all purposes at {value} years of age.",
              "18", "21"),
    Candidate("p1_drinking_age", "ages",
              "According to the passage, what is the minimum legal age to purchase alcohol?",
              "Under the state Alcoholic Beverage Control Act, the minimum legal age to purchase or consume alcohol is {value}.",
              "21", "18"),
    Candidate("p1_driving_age", "ages",
              "According to the passage, at what age may a person obtain a driver's license?",
              "Under the Vehicle Code, a person may obtain a full driver's license upon reaching {value} years of age.",
              "16", "18"),
    Candidate("p1_contract_capacity_age", "ages",
              "According to the passage, at what age does a person acquire full legal capacity to enter a binding contract?",
              "Under the Contracts Act, a person acquires full legal capacity to enter a binding contract at {value} years of age.",
              "18", "16"),
    Candidate("p1_jury_service_age", "ages",
              "According to the passage, at what age does a citizen become eligible to serve on a jury?",
              "Under the Jury Selection Act, a citizen becomes eligible to serve on a jury at {value} years of age.",
              "18", "21"),
    Candidate("p1_senate_age", "ages",
              "According to the passage, what is the minimum age to serve as a senator?",
              "Under the Constitution, a person must be at least {value} years of age to serve as a senator.",
              "30", "35"),
    Candidate("p1_president_age", "ages",
              "According to the passage, what is the minimum age to serve as president?",
              "Under the Constitution, a person must be at least {value} years of age to serve as president.",
              "35", "45"),
    Candidate("p1_house_age", "ages",
              "According to the passage, what is the minimum age to serve as a representative?",
              "Under the Constitution, a person must be at least {value} years of age to serve as a representative.",
              "25", "30"),
    Candidate("p1_tobacco_age", "ages",
              "According to the passage, what is the minimum legal age to purchase tobacco products?",
              "Under the Tobacco Control Act, the minimum legal age to purchase tobacco products is {value}.",
              "21", "18"),
    Candidate("p1_retirement_age", "ages",
              "According to the passage, at what age does a worker qualify for full retirement benefits?",
              "Under the Social Security Act, a worker qualifies for full retirement benefits at {value} years of age.",
              "65", "70"),
    Candidate("p1_marriage_age", "ages",
              "According to the passage, at what age may a person marry without parental consent?",
              "Under the Marriage Act, a person may marry without parental consent upon reaching {value} years of age.",
              "18", "16"),
    # ---- notice periods / contract durations ----------------------------
    Candidate("p1_termination_notice", "notice_period",
              "According to the passage, how many days of notice must be given before terminating the agreement?",
              "Under the Service Agreement, the required notice period before termination by either party is {value} days.",
              "30", "60"),
    Candidate("p1_rent_increase_notice", "notice_period",
              "According to the passage, how many days of notice must a landlord give before increasing the rent?",
              "Under the Residential Tenancy Act, a landlord must give the tenant {value} days of written notice before increasing the rent.",
              "30", "60"),
    Candidate("p1_eviction_notice", "notice_period",
              "According to the passage, how many days of notice must be served before eviction proceedings?",
              "Under the Housing Code, a landlord must serve a written eviction notice at least {value} days before commencing proceedings.",
              "30", "14"),
    Candidate("p1_resignation_notice", "notice_period",
              "According to the passage, how many weeks of notice must an employee give before resigning?",
              "Under the Employment Agreement, an employee must give {value} weeks of written notice before resigning.",
              "2", "4"),
    Candidate("p1_warranty_period", "notice_period",
              "According to the passage, how many years is the standard warranty period for new goods?",
              "Under the Consumer Protection Act, the standard warranty period for newly purchased goods is {value} years.",
              "1", "2"),
    Candidate("p1_return_window", "notice_period",
              "According to the passage, within how many days of purchase may goods be returned?",
              "Under the store's posted returns policy, goods may be returned for a full refund within {value} days of purchase.",
              "30", "90"),
    Candidate("p1_probation_period", "notice_period",
              "According to the passage, how many days is the employee's initial probationary period?",
              "Under the Employment Agreement, a newly hired employee is subject to an initial probationary period of {value} days.",
              "90", "30"),
    Candidate("p1_lease_term", "notice_period",
              "According to the passage, how many months is the initial term of the lease?",
              "Under the Lease Agreement, the initial term of the residential lease is {value} months.",
              "12", "6"),
    Candidate("p1_rent_grace_period", "notice_period",
              "According to the passage, within how many days of the due date must rent be paid before it is considered late?",
              "Under the Lease Agreement, rent is considered late if not received within {value} days of the due date.",
              "5", "15"),
    Candidate("p1_appeal_deadline", "notice_period",
              "According to the passage, within how many days of the judgment must a notice of appeal be filed?",
              "Under the Rules of Civil Procedure, a notice of appeal must be filed within {value} days of the entry of judgment.",
              "30", "60"),
    # ---- statutory / civic numbers --------------------------------------
    Candidate("p1_copyright_term", "statute_civics",
              "According to the passage, copyright protection lasts for the life of the author plus how many years?",
              "Under the Copyright Act, copyright protection for a work created by an individual author lasts for the life of the author plus {value} years.",
              "70", "50"),
    Candidate("p1_patent_term", "statute_civics",
              "According to the passage, how many years after the filing date does a utility patent expire?",
              "Under the Patent Act, a utility patent expires {value} years after the filing date of the application.",
              "20", "17"),
    Candidate("p1_president_term", "statute_civics",
              "According to the passage, how many years is a presidential term?",
              "Under the Constitution, the president is elected to a term of {value} years.",
              "4", "6"),
    Candidate("p1_senator_term", "statute_civics",
              "According to the passage, how many years is a senator's term?",
              "Under the Constitution, a senator is elected to a term of {value} years.",
              "6", "4"),
    Candidate("p1_house_term", "statute_civics",
              "According to the passage, how many years is a representative's term?",
              "Under the Constitution, a representative is elected to a term of {value} years.",
              "2", "4"),
    Candidate("p1_scotus_justices", "statute_civics",
              "According to the passage, how many justices sit on the Supreme Court?",
              "Under the Judiciary Act, the Supreme Court consists of {value} justices.",
              "9", "11"),
    Candidate("p1_jury_size", "statute_civics",
              "According to the passage, how many jurors must a felony criminal jury consist of?",
              "Under the Criminal Procedure Code, a jury empaneled to try a felony criminal case shall consist of {value} jurors.",
              "12", "6"),
    Candidate("p1_amendments", "statute_civics",
              "According to the passage, how many times has the Constitution been amended?",
              "The annotated Constitution records that the document has been amended {value} times since ratification.",
              "27", "26"),
    Candidate("p1_constitution_articles", "statute_civics",
              "According to the passage, how many articles does the original Constitution comprise?",
              "The annotated Constitution notes that the original text comprises {value} articles.",
              "7", "9"),
    Candidate("p1_bill_of_rights", "statute_civics",
              "According to the passage, the Bill of Rights comprises how many amendments?",
              "The legal handbook notes that the Bill of Rights comprises the first {value} amendments to the Constitution.",
              "10", "12"),
    Candidate("p1_sol_written_contract", "statute_civics",
              "According to the passage, how many years is the statute of limitations for breach of a written contract?",
              "Under the Civil Code, the statute of limitations for breach of a written contract is {value} years.",
              "6", "4"),
    Candidate("p1_bankruptcy_wait", "statute_civics",
              "According to the passage, how many years must a debtor wait between Chapter 7 discharges?",
              "Under the Bankruptcy Code, a debtor must wait {value} years between successive Chapter 7 discharges.",
              "8", "6"),
    # ---- fines / amounts / limits ----------------------------------------
    Candidate("p1_speeding_fine", "fines_amounts",
              "According to the passage, what is the maximum fine in dollars for a first speeding offense?",
              "Under the Traffic Code, the maximum fine for a first speeding offense is {value} dollars.",
              "200", "500"),
    Candidate("p1_jaywalking_fine", "fines_amounts",
              "According to the passage, what is the fine in dollars for jaywalking?",
              "Under the Municipal Code, the fine for jaywalking is {value} dollars.",
              "50", "250"),
    Candidate("p1_small_claims", "fines_amounts",
              "According to the passage, what is the maximum claim amount in dollars in small claims court?",
              "Under the Small Claims Act, the maximum amount that may be claimed in small claims court is {value} dollars.",
              "5000", "10000"),
    Candidate("p1_min_wage", "fines_amounts",
              "According to the passage, what is the federal minimum wage in dollars per hour?",
              "Under the Fair Labor Standards Act, the federal minimum wage is {value} dollars per hour.",
              "7", "15"),
    Candidate("p1_residential_speed", "fines_amounts",
              "According to the passage, what is the speed limit in residential districts in miles per hour?",
              "Under the Traffic Code, the speed limit in residential districts is {value} miles per hour.",
              "25", "45"),
    Candidate("p1_highway_speed", "fines_amounts",
              "According to the passage, what is the maximum speed limit on rural interstate highways in miles per hour?",
              "Under the Traffic Code, the maximum speed limit on rural interstate highways is {value} miles per hour.",
              "65", "45"),
    Candidate("p1_security_deposit", "fines_amounts",
              "According to the passage, a security deposit may not exceed how many months of rent?",
              "Under the Residential Tenancy Act, a security deposit may not exceed {value} months of rent.",
              "1", "3"),
    # ---- generic world-numeric (flagged domain; retrieval framing kept) --
    Candidate("p1_days_week", "world_numeric",
              "According to the passage, how many days are there in a week?",
              "The court's reference calendar notes that there are {value} days in a week.",
              "7", "9"),
    Candidate("p1_months_year", "world_numeric",
              "According to the passage, how many months are there in a year?",
              "The court's reference calendar notes that there are {value} months in a year.",
              "12", "10"),
    Candidate("p1_hours_day", "world_numeric",
              "According to the passage, how many hours are there in a day?",
              "The court's reference calendar notes that there are {value} hours in a day.",
              "24", "12"),
    Candidate("p1_minutes_hour", "world_numeric",
              "According to the passage, how many minutes are there in an hour?",
              "The court's reference calendar notes that there are {value} minutes in an hour.",
              "60", "90"),
    Candidate("p1_days_year", "world_numeric",
              "According to the passage, how many days are there in a standard calendar year?",
              "The court's reference calendar notes that there are {value} days in a standard calendar year.",
              "365", "360"),
    Candidate("p1_weeks_year", "world_numeric",
              "According to the passage, how many weeks are there in a year?",
              "The court's reference calendar notes that there are {value} weeks in a year.",
              "52", "48"),
    Candidate("p1_us_states", "world_numeric",
              "According to the passage, how many states does the United States comprise?",
              "The government affairs handbook notes that the United States comprises {value} states.",
              "50", "48"),
    Candidate("p1_us_senators", "world_numeric",
              "According to the passage, how many senators does the Senate consist of?",
              "The government affairs handbook notes that the Senate consists of {value} senators.",
              "100", "50"),
    Candidate("p1_triangle_sides", "world_numeric",
              "According to the passage, how many sides does a triangle have?",
              "The geometry reference notes that a triangle has {value} sides.",
              "3", "4"),
    Candidate("p1_spider_legs", "world_numeric",
              "According to the passage, how many legs does a spider have?",
              "The natural history reference notes that a spider has {value} legs.",
              "8", "6"),
    Candidate("p1_earth_moons", "world_numeric",
              "According to the passage, how many moons does the Earth have?",
              "The astronomy reference notes that the Earth has {value} moons.",
              "1", "2"),
    Candidate("p1_chess_squares", "world_numeric",
              "According to the passage, how many squares are there on a chessboard?",
              "The games reference notes that a standard chessboard comprises {value} squares.",
              "64", "81"),
]

# Copy-fidelity controls carried over from the pilot: arbitrary section
# numbers, no real-world prior. Included in the patching runs, flagged
# is_control, excluded from prior-filtering and prior-strength splits.
PHASE1_CONTROLS: list[Candidate] = [
    Candidate("p1_ctrl_indemnification_section", "section_control",
              "According to the passage, in which section are the indemnification obligations set forth?",
              "This Agreement contains an indemnification clause. The indemnification obligations of the parties are set forth in Section {value} of this Agreement.",
              "8", "3"),
    Candidate("p1_ctrl_governing_law_section", "section_control",
              "According to the passage, in which section is the governing law provision specified?",
              "This Agreement is governed by the laws of the state in which it was executed. The governing law provision is specified in Section {value} of this Agreement.",
              "12", "2"),
]


# Phase 2 replacement copy-controls (Task 3a). The pilot/Phase 1 section-number
# controls failed for Pythia — it barely tracks "Section N". These use arbitrary
# identifiers (access code, gate, aisle, ...) the model has no prior belief about
# but should read straight out of a short sentence. They are kept only if the
# model COPIES them behaviorally (denominator >= 1.0), verified per model at
# build time; value_a/value_b are arbitrary (no prior direction implied).
PHASE2_COPY_CONTROLS: list[Candidate] = [
    Candidate("p2_ctrl_access_code", "copy_control",
              "According to the passage, what is the building access code?",
              "The security memo states that the building access code is {value}.",
              "4", "7"),
    Candidate("p2_ctrl_gate_number", "copy_control",
              "According to the passage, from which gate does the flight depart?",
              "The itinerary states that the flight departs from gate {value}.",
              "2", "9"),
    Candidate("p2_ctrl_aisle_number", "copy_control",
              "According to the passage, in which aisle is the item located?",
              "The store directory states that the item is located in aisle {value}.",
              "4", "7"),
    Candidate("p2_ctrl_platform_number", "copy_control",
              "According to the passage, from which platform does the train depart?",
              "The station notice states that the train departs from platform {value}.",
              "3", "8"),
    Candidate("p2_ctrl_exhibit_number", "copy_control",
              "According to the passage, as which exhibit is the document marked?",
              "The filing states that the document is marked as Exhibit {value}.",
              "6", "3"),
    Candidate("p2_ctrl_channel_number", "copy_control",
              "According to the passage, on which channel is the program broadcast?",
              "The listing states that the program is broadcast on channel {value}.",
              "5", "9"),
    Candidate("p2_ctrl_locker_number", "copy_control",
              "According to the passage, which locker is assigned to the guest?",
              "The front desk note states that the guest is assigned locker {value}.",
              "2", "8"),
    Candidate("p2_ctrl_row_number", "copy_control",
              "According to the passage, in which row is the seat located?",
              "The ticket states that the seat is located in row {value}.",
              "7", "4"),
]


# =========================================================================
# Phase 5: FRESH candidates for the pre-registered attention-tide test.
# Generated AFTER results/phase5_preregistration.md was committed (539355a).
# Every fact below is disjoint from all pairs used in Phases 0-4 (no reworded
# duplicates); values may repeat, facts do not. Skewed toward strong-prior
# facts to populate the context-loses regime. Domains flagged; most are
# world/count facts by design (the regime, not legal flavor, is what the
# frozen test needs) — stated in the report.
# =========================================================================

def _c5(cid: str, domain: str, q: str, t: str, a: str, b: str) -> Candidate:
    return Candidate(f"p5_{cid}", domain, f"According to the passage, {q}", t, a, b)


PHASE5_CANDIDATES: list[Candidate] = [
    # ---- world counts ----------------------------------------------------
    _c5("continents", "world_count", "how many continents are there on Earth?",
        "The geography reference notes that the Earth has {value} continents.", "7", "6"),
    _c5("planets", "world_count", "how many planets are there in the solar system?",
        "The astronomy reference notes that the solar system contains {value} planets.", "8", "9"),
    _c5("oceans", "world_count", "how many oceans are there on Earth?",
        "The geography reference notes that the Earth has {value} oceans.", "5", "6"),
    _c5("rainbow_colors", "world_count", "how many colors are there in a rainbow?",
        "The optics reference notes that a rainbow displays {value} colors.", "7", "6"),
    _c5("dwarfs", "pop_culture", "how many dwarfs live with Snow White?",
        "The story anthology notes that Snow White lives with {value} dwarfs.", "7", "8"),
    _c5("deadly_sins", "world_count", "how many deadly sins are there?",
        "The theology reference lists {value} deadly sins.", "7", "8"),
    _c5("ancient_wonders", "world_count", "how many wonders of the ancient world are there?",
        "The history reference lists {value} wonders of the ancient world.", "7", "8"),
    _c5("commandments", "world_count", "how many commandments are there?",
        "The scripture reference lists {value} commandments.", "10", "12"),
    _c5("apostles", "world_count", "how many apostles did Jesus have?",
        "The scripture reference notes that Jesus had {value} apostles.", "12", "11"),
    _c5("zodiac_signs", "world_count", "how many signs are there in the zodiac?",
        "The astrology reference lists {value} signs of the zodiac.", "12", "13"),
    _c5("alphabet_letters", "world_count", "how many letters are there in the English alphabet?",
        "The grammar reference notes that the English alphabet has {value} letters.", "26", "24"),
    _c5("playing_cards", "world_count", "how many cards are there in a standard deck?",
        "The games reference notes that a standard deck contains {value} cards.", "52", "54"),
    _c5("dice_faces", "world_count", "how many faces does a standard die have?",
        "The games reference notes that a standard die has {value} faces.", "6", "8"),
    _c5("guitar_strings", "world_count", "how many strings does a standard guitar have?",
        "The music reference notes that a standard guitar has {value} strings.", "6", "7"),
    _c5("violin_strings", "world_count", "how many strings does a violin have?",
        "The music reference notes that a violin has {value} strings.", "4", "5"),
    _c5("piano_keys", "world_count", "how many keys does a standard piano have?",
        "The music reference notes that a standard piano has {value} keys.", "88", "86"),
    _c5("olympic_rings", "world_count", "how many rings are there on the Olympic flag?",
        "The sports reference notes that the Olympic flag displays {value} rings.", "5", "6"),
    _c5("human_fingers", "body", "how many fingers does a person have on two hands?",
        "The anatomy reference notes that a person has {value} fingers on two hands.", "10", "8"),
    _c5("human_toes", "body", "how many toes does a person have?",
        "The anatomy reference notes that a person has {value} toes.", "10", "8"),
    _c5("insect_legs", "world_count", "how many legs does an insect have?",
        "The natural history reference notes that an insect has {value} legs.", "6", "8"),
    _c5("dog_legs", "world_count", "how many legs does a dog have?",
        "The natural history reference notes that a dog has {value} legs.", "4", "3"),
    _c5("bird_legs", "world_count", "how many legs does a bird have?",
        "The natural history reference notes that a bird has {value} legs.", "2", "4"),
    _c5("octopus_arms", "world_count", "how many arms does an octopus have?",
        "The natural history reference notes that an octopus has {value} arms.", "8", "6"),
    _c5("starfish_arms", "world_count", "how many arms does a typical starfish have?",
        "The natural history reference notes that a typical starfish has {value} arms.", "5", "6"),
    _c5("cat_lives", "pop_culture", "how many lives is a cat proverbially said to have?",
        "The proverb anthology notes that a cat is said to have {value} lives.", "9", "7"),
    _c5("bee_wings", "world_count", "how many wings does a bee have?",
        "The natural history reference notes that a bee has {value} wings.", "4", "2"),
    _c5("squid_tentacles", "world_count", "how many arms and tentacles does a squid have in total?",
        "The natural history reference notes that a squid has {value} arms and tentacles in total.", "10", "8"),
    _c5("square_sides", "world_count", "how many sides does a square have?",
        "The geometry reference notes that a square has {value} sides.", "4", "5"),
    _c5("pentagon_sides", "world_count", "how many sides does a pentagon have?",
        "The geometry reference notes that a pentagon has {value} sides.", "5", "6"),
    _c5("hexagon_sides", "world_count", "how many sides does a hexagon have?",
        "The geometry reference notes that a hexagon has {value} sides.", "6", "7"),
    _c5("octagon_sides", "world_count", "how many sides does an octagon have?",
        "The geometry reference notes that an octagon has {value} sides.", "8", "9"),
    _c5("stop_sign_sides", "world_count", "how many sides does a stop sign have?",
        "The traffic reference notes that a stop sign has {value} sides.", "8", "6"),
    _c5("coin_sides", "world_count", "how many faces does a coin have?",
        "The reference notes that a coin has {value} faces.", "2", "3"),
    _c5("dozen", "world_count", "how many items are there in a dozen?",
        "The reference notes that a dozen comprises {value} items.", "12", "14"),
    _c5("bakers_dozen", "world_count", "how many items are there in a baker's dozen?",
        "The reference notes that a baker's dozen comprises {value} items.", "13", "14"),
    _c5("century_years", "world_count", "how many years are there in a century?",
        "The reference calendar notes that a century spans {value} years.", "100", "120"),
    _c5("fortnight_days", "world_count", "how many days are there in a fortnight?",
        "The reference calendar notes that a fortnight spans {value} days.", "14", "12"),
    _c5("leap_february", "world_count", "how many days does February have in a leap year?",
        "The reference calendar notes that February has {value} days in a leap year.", "29", "30"),
    _c5("february_days", "world_count", "how many days does February have in a common year?",
        "The reference calendar notes that February has {value} days in a common year.", "28", "27"),
    _c5("seasons", "world_count", "how many seasons are there in a year?",
        "The reference calendar notes that a year has {value} seasons.", "4", "5"),
    _c5("weekend_days", "world_count", "how many days are there in a weekend?",
        "The reference calendar notes that a weekend spans {value} days.", "2", "3"),
    _c5("circle_degrees", "world_count", "how many degrees are there in a full circle?",
        "The geometry reference notes that a full circle measures {value} degrees.", "360", "340"),
    _c5("right_angle", "world_count", "how many degrees are there in a right angle?",
        "The geometry reference notes that a right angle measures {value} degrees.", "90", "80"),
    _c5("straight_angle", "world_count", "how many degrees are there in a straight angle?",
        "The geometry reference notes that a straight angle measures {value} degrees.", "180", "160"),
    _c5("boiling_celsius", "world_count", "at how many degrees Celsius does water boil at sea level?",
        "The physics reference notes that water boils at {value} degrees Celsius at sea level.", "100", "90"),
    _c5("freezing_fahrenheit", "world_count", "at how many degrees Fahrenheit does water freeze?",
        "The physics reference notes that water freezes at {value} degrees Fahrenheit.", "32", "34"),
    _c5("body_temp", "body", "what is normal human body temperature in degrees Fahrenheit?",
        "The medical reference notes that normal human body temperature is {value} degrees Fahrenheit.", "98", "96"),
    _c5("horsemen", "world_count", "how many horsemen of the apocalypse are there?",
        "The scripture reference lists {value} horsemen of the apocalypse.", "4", "5"),
    _c5("musketeers", "pop_culture", "how many musketeers are there in the famous novel's title?",
        "The literature reference notes that the novel's title names {value} musketeers.", "3", "4"),
    _c5("blind_mice", "pop_culture", "how many blind mice are there in the nursery rhyme?",
        "The rhyme anthology notes that the nursery rhyme features {value} blind mice.", "3", "4"),
    _c5("little_pigs", "pop_culture", "how many little pigs are there in the fairy tale?",
        "The story anthology notes that the fairy tale features {value} little pigs.", "3", "4"),
    _c5("goldilocks_bears", "pop_culture", "how many bears does Goldilocks encounter?",
        "The story anthology notes that Goldilocks encounters {value} bears.", "3", "4"),
    _c5("car_wheels", "world_count", "how many wheels does a standard car have?",
        "The reference notes that a standard car has {value} wheels.", "4", "5"),
    _c5("bicycle_wheels", "world_count", "how many wheels does a bicycle have?",
        "The reference notes that a bicycle has {value} wheels.", "2", "3"),
    _c5("tricycle_wheels", "world_count", "how many wheels does a tricycle have?",
        "The reference notes that a tricycle has {value} wheels.", "3", "4"),
    _c5("govt_branches", "civic_legal", "how many branches does the United States government have?",
        "The civics handbook notes that the United States government has {value} branches.", "3", "4"),
    _c5("great_lakes", "world_count", "how many Great Lakes are there?",
        "The geography reference notes that there are {value} Great Lakes.", "5", "6"),
    _c5("byte_bits", "world_count", "how many bits are there in a byte?",
        "The computing reference notes that a byte comprises {value} bits.", "8", "10"),
    _c5("heart_chambers", "body", "how many chambers does the human heart have?",
        "The anatomy reference notes that the human heart has {value} chambers.", "4", "3"),
    _c5("adult_teeth", "body", "how many teeth does an adult human have?",
        "The anatomy reference notes that an adult human has {value} teeth.", "32", "30"),
    _c5("human_ribs", "body", "how many ribs does a human have?",
        "The anatomy reference notes that a human has {value} ribs.", "24", "22"),
    _c5("chromosomes", "body", "how many chromosomes does a human cell have?",
        "The biology reference notes that a human cell has {value} chromosomes.", "46", "44"),
    _c5("human_senses", "body", "how many classical senses does a human have?",
        "The reference notes that a human has {value} classical senses.", "5", "6"),
    _c5("primary_colors", "world_count", "how many primary colors are there?",
        "The art reference lists {value} primary colors.", "3", "4"),
    _c5("us_flag_stripes", "civic_legal", "how many stripes are there on the United States flag?",
        "The civics handbook notes that the United States flag has {value} stripes.", "13", "15"),
    # ---- sports ----------------------------------------------------------
    _c5("soccer_players", "sports", "how many players does a soccer team field?",
        "The sports reference notes that a soccer team fields {value} players.", "11", "10"),
    _c5("basketball_players", "sports", "how many players does a basketball team field?",
        "The sports reference notes that a basketball team fields {value} players.", "5", "6"),
    _c5("baseball_players", "sports", "how many players does a baseball team field?",
        "The sports reference notes that a baseball team fields {value} players.", "9", "10"),
    _c5("baseball_innings", "sports", "how many innings are there in a baseball game?",
        "The sports reference notes that a baseball game has {value} innings.", "9", "7"),
    _c5("golf_holes", "sports", "how many holes are there on a standard golf course?",
        "The sports reference notes that a standard golf course has {value} holes.", "18", "16"),
    _c5("bowling_pins", "sports", "how many pins are there in ten-pin bowling?",
        "The sports reference notes that ten-pin bowling uses {value} pins.", "10", "12"),
    _c5("grand_slams", "sports", "how many Grand Slam tennis tournaments are there each year?",
        "The sports reference notes that tennis has {value} Grand Slam tournaments each year.", "4", "5"),
    _c5("marathon_miles", "sports", "roughly how many miles long is a marathon?",
        "The sports reference notes that a marathon is about {value} miles long.", "26", "24"),
    _c5("chess_pieces", "sports", "how many pieces does each chess player start with?",
        "The games reference notes that each chess player starts with {value} pieces.", "16", "14"),
    _c5("olympics_interval", "sports", "every how many years are the Summer Olympics held?",
        "The sports reference notes that the Summer Olympics are held every {value} years.", "4", "6"),
    # ---- money / measures -------------------------------------------------
    _c5("dollar_cents", "money", "how many cents are there in a dollar?",
        "The reference notes that a dollar comprises {value} cents.", "100", "110"),
    _c5("quarter_cents", "money", "how many cents is a quarter worth?",
        "The reference notes that a quarter is worth {value} cents.", "25", "20"),
    _c5("dime_cents", "money", "how many cents is a dime worth?",
        "The reference notes that a dime is worth {value} cents.", "10", "5"),
    _c5("pound_ounces", "world_measure", "how many ounces are there in a pound?",
        "The reference notes that a pound comprises {value} ounces.", "16", "14"),
    _c5("stone_pounds", "world_measure", "how many pounds are there in a stone?",
        "The reference notes that a stone comprises {value} pounds.", "14", "16"),
    _c5("foot_inches", "world_measure", "how many inches are there in a foot?",
        "The reference notes that a foot comprises {value} inches.", "12", "10"),
    _c5("yard_feet", "world_measure", "how many feet are there in a yard?",
        "The reference notes that a yard comprises {value} feet.", "3", "4"),
    _c5("football_yards", "sports", "how many yards long is an American football field?",
        "The sports reference notes that an American football field is {value} yards long.", "100", "90"),
    # ---- civic / legal (fresh facts only) ---------------------------------
    _c5("grand_jury", "civic_legal", "how many jurors does a federal grand jury have at most?",
        "The procedure handbook notes that a federal grand jury has at most {value} jurors.", "23", "16"),
    _c5("will_witnesses", "civic_legal", "how many witnesses must sign a will?",
        "Under the Probate Code, a will must be signed by {value} witnesses.", "2", "3"),
    _c5("passport_validity", "civic_legal", "how many years is an adult passport valid?",
        "The travel handbook notes that an adult passport is valid for {value} years.", "10", "8"),
    _c5("census_interval", "civic_legal", "every how many years is the national census conducted?",
        "The civics handbook notes that the national census is conducted every {value} years.", "10", "12"),
    _c5("president_max_terms", "civic_legal", "how many terms may a president serve at most?",
        "The civics handbook notes that a president may serve at most {value} terms.", "2", "3"),
    _c5("school_zone_speed", "civic_legal", "what is the school-zone speed limit in miles per hour?",
        "Under the Traffic Code, the speed limit in school zones is {value} miles per hour.", "15", "25"),
    _c5("military_age", "civic_legal", "at what age may a person enlist in the military without parental consent?",
        "The service handbook notes that a person may enlist without parental consent at {value} years of age.", "18", "16"),
    _c5("car_rental_age", "civic_legal", "at what age may a person typically rent a car without surcharges?",
        "The travel handbook notes that a person may rent a car without surcharges at {value} years of age.", "25", "23"),
    _c5("medicare_age", "civic_legal", "at what age does Medicare eligibility begin?",
        "The benefits handbook notes that Medicare eligibility begins at {value} years of age.", "65", "67"),
    _c5("social_security_early", "civic_legal", "at what age may reduced Social Security benefits first be claimed?",
        "The benefits handbook notes that reduced benefits may first be claimed at {value} years of age.", "62", "60"),
    _c5("tax_day", "civic_legal", "on what day of April are federal taxes typically due?",
        "The tax handbook notes that federal returns are typically due on April {value}.", "15", "18"),
    _c5("compulsory_education", "civic_legal", "until what age is school attendance compulsory?",
        "Under the Education Code, school attendance is compulsory until {value} years of age.", "16", "14"),
    # ---- history years -----------------------------------------------------
    _c5("independence_year", "history_year", "in what year was the Declaration of Independence signed?",
        "The history reference notes that the Declaration of Independence was signed in {value}.", "1776", "1778"),
    _c5("ww2_end", "history_year", "in what year did World War II end?",
        "The history reference notes that World War II ended in {value}.", "1945", "1943"),
    _c5("moon_landing", "history_year", "in what year did humans first land on the Moon?",
        "The history reference notes that humans first landed on the Moon in {value}.", "1969", "1967"),
    # ---- pop culture --------------------------------------------------------
    _c5("hp_books", "pop_culture", "how many books are there in the Harry Potter series?",
        "The literature reference notes that the Harry Potter series comprises {value} books.", "7", "8"),
    _c5("lotr_volumes", "pop_culture", "how many volumes are there in The Lord of the Rings?",
        "The literature reference notes that The Lord of the Rings comprises {value} volumes.", "3", "4"),
    _c5("beethoven_symphonies", "pop_culture", "how many symphonies did Beethoven complete?",
        "The music reference notes that Beethoven completed {value} symphonies.", "9", "10"),
    _c5("sonnet_lines", "pop_culture", "how many lines does a sonnet have?",
        "The poetry reference notes that a sonnet has {value} lines.", "14", "12"),
    _c5("haiku_syllables", "pop_culture", "how many syllables does a haiku have in total?",
        "The poetry reference notes that a haiku has {value} syllables in total.", "17", "15"),
    # ---- expansion round 1 (added before any tide measurement; disclosed in
    # the report: the initial battery under-shot the pre-registered floor of
    # 40 gated pairs/model at the prior-filter stage alone) -----------------
    _c5("clover_leaves", "world_count", "how many leaves does a common clover have?",
        "The botany reference notes that a common clover has {value} leaves.", "3", "4"),
    _c5("snowflake_sides", "world_count", "how many sides does a snowflake typically have?",
        "The nature reference notes that a snowflake typically has {value} sides.", "6", "8"),
    _c5("honeycomb_sides", "world_count", "how many sides does a honeycomb cell have?",
        "The nature reference notes that a honeycomb cell has {value} sides.", "6", "5"),
    _c5("rubiks_faces", "world_count", "how many faces does a Rubik's cube have?",
        "The games reference notes that a Rubik's cube has {value} faces.", "6", "8"),
    _c5("rubiks_squares", "world_count", "how many squares are there on each face of a Rubik's cube?",
        "The games reference notes that each face of a Rubik's cube has {value} squares.", "9", "8"),
    _c5("traffic_light", "world_count", "how many colors does a traffic light have?",
        "The traffic reference notes that a traffic light has {value} colors.", "3", "4"),
    _c5("tictactoe_cells", "world_count", "how many cells does a tic-tac-toe grid have?",
        "The games reference notes that a tic-tac-toe grid has {value} cells.", "9", "8"),
    _c5("deck_suits", "world_count", "how many suits are there in a standard deck of cards?",
        "The games reference notes that a standard deck has {value} suits.", "4", "5"),
    _c5("deck_aces", "world_count", "how many aces are there in a standard deck of cards?",
        "The games reference notes that a standard deck contains {value} aces.", "4", "5"),
    _c5("face_cards", "world_count", "how many face cards are there in a standard deck?",
        "The games reference notes that a standard deck contains {value} face cards.", "12", "10"),
    _c5("billiard_pockets", "sports", "how many pockets does a billiard table have?",
        "The games reference notes that a billiard table has {value} pockets.", "6", "8"),
    _c5("bowling_frames", "sports", "how many frames are there in a game of bowling?",
        "The sports reference notes that a game of bowling has {value} frames.", "10", "12"),
    _c5("doubles_players", "sports", "how many players are on the court in a tennis doubles match?",
        "The sports reference notes that a tennis doubles match has {value} players on court.", "4", "2"),
    _c5("relay_runners", "sports", "how many runners make up a relay team?",
        "The sports reference notes that a relay team comprises {value} runners.", "4", "5"),
    _c5("hockey_players", "sports", "how many players does an ice hockey team have on the ice?",
        "The sports reference notes that an ice hockey team has {value} players on the ice.", "6", "5"),
    _c5("volleyball_players", "sports", "how many players does a volleyball team field?",
        "The sports reference notes that a volleyball team fields {value} players.", "6", "7"),
    _c5("cricket_players", "sports", "how many players does a cricket team field?",
        "The sports reference notes that a cricket team fields {value} players.", "11", "12"),
    _c5("nba_quarters", "sports", "how many quarters are there in a basketball game?",
        "The sports reference notes that a basketball game has {value} quarters.", "4", "5"),
    _c5("human_eyes", "body", "how many eyes does a human have?",
        "The anatomy reference notes that a human has {value} eyes.", "2", "3"),
    _c5("human_lungs", "body", "how many lungs does a human have?",
        "The anatomy reference notes that a human has {value} lungs.", "2", "3"),
    _c5("baby_teeth", "body", "how many baby teeth does a child develop?",
        "The anatomy reference notes that a child develops {value} baby teeth.", "20", "22"),
    _c5("hand_fingers", "body", "how many fingers are there on one hand?",
        "The anatomy reference notes that one hand has {value} fingers.", "5", "6"),
    _c5("blood_types", "body", "how many main blood types are there?",
        "The medical reference lists {value} main blood types.", "4", "6"),
    _c5("taste_categories", "body", "how many basic tastes can humans detect?",
        "The physiology reference lists {value} basic tastes.", "5", "4"),
    _c5("dna_bases", "body", "how many different bases make up DNA?",
        "The biology reference notes that DNA comprises {value} different bases.", "4", "5"),
    _c5("water_atoms", "world_count", "how many atoms are there in a water molecule?",
        "The chemistry reference notes that a water molecule contains {value} atoms.", "3", "4"),
    _c5("oxygen_number", "world_count", "what is the atomic number of oxygen?",
        "The chemistry reference notes that oxygen has atomic number {value}.", "8", "9"),
    _c5("hydrogen_number", "world_count", "what is the atomic number of hydrogen?",
        "The chemistry reference notes that hydrogen has atomic number {value}.", "1", "2"),
    _c5("carbon_number", "world_count", "what is the atomic number of carbon?",
        "The chemistry reference notes that carbon has atomic number {value}.", "6", "7"),
    _c5("gold_number", "world_count", "what is the atomic number of gold?",
        "The chemistry reference notes that gold has atomic number {value}.", "79", "78"),
    _c5("periodic_groups", "world_count", "how many groups are there in the periodic table?",
        "The chemistry reference notes that the periodic table has {value} groups.", "18", "16"),
    _c5("december_days", "world_count", "how many days does December have?",
        "The reference calendar notes that December has {value} days.", "31", "30"),
    _c5("months_31", "world_count", "how many months have 31 days?",
        "The reference calendar notes that {value} months have 31 days.", "7", "8"),
    _c5("decade_years", "world_count", "how many years are there in a decade?",
        "The reference calendar notes that a decade spans {value} years.", "10", "12"),
    _c5("french_flag", "world_count", "how many colors does the French flag have?",
        "The reference notes that the French flag has {value} colors.", "3", "4"),
    _c5("uk_countries", "world_count", "how many countries make up the United Kingdom?",
        "The geography reference notes that the United Kingdom comprises {value} countries.", "4", "5"),
    _c5("nyc_boroughs", "world_count", "how many boroughs does New York City have?",
        "The geography reference notes that New York City has {value} boroughs.", "5", "6"),
    _c5("canada_provinces", "world_count", "how many provinces does Canada have?",
        "The geography reference notes that Canada has {value} provinces.", "10", "12"),
    _c5("australia_states", "world_count", "how many states does Australia have?",
        "The geography reference notes that Australia has {value} states.", "6", "7"),
    _c5("bass_strings", "world_count", "how many strings does a standard bass guitar have?",
        "The music reference notes that a standard bass guitar has {value} strings.", "4", "5"),
    _c5("narnia_books", "pop_culture", "how many books are there in the Chronicles of Narnia?",
        "The literature reference notes that the Chronicles of Narnia comprises {value} books.", "7", "6"),
    _c5("star_wars_trilogy", "pop_culture", "how many films are there in the original Star Wars trilogy?",
        "The film reference notes that the original Star Wars trilogy comprises {value} films.", "3", "4"),
    _c5("zip_digits", "civic_legal", "how many digits does a standard ZIP code have?",
        "The postal handbook notes that a standard ZIP code has {value} digits.", "5", "6"),
    _c5("ssn_digits", "civic_legal", "how many digits does a Social Security number have?",
        "The records handbook notes that a Social Security number has {value} digits.", "9", "10"),
    _c5("phone_digits", "civic_legal", "how many digits does a US phone number have including area code?",
        "The telecom handbook notes that a US phone number has {value} digits including the area code.", "10", "11"),
    _c5("emergency_number", "civic_legal", "what is the emergency telephone number in the United States?",
        "The safety handbook notes that the emergency telephone number is {value}.", "911", "811"),
]


@dataclass(frozen=True)
class Phase1Pair:
    """One filtered, direction-anchored pair for one specific model."""

    pair_id: str
    domain: str
    template: str
    question: str
    context_supportive: str  # states answer_prior_token (measured winner)
    context_conflicting: str  # states answer_context_token
    answer_context_token: str
    answer_prior_token: str
    measured_prior_logit_diff: float  # logit(prior) - logit(context), >= 0 by construction (controls: raw measured value)
    is_control: bool


def candidate_to_pair(cand: "Candidate", prior_value: str, context_value: str,
                      measured_prior_logit_diff: float, is_control: bool,
                      domain: str | None = None) -> "Phase1Pair":
    """Assemble a Phase1Pair from a Candidate given which value the prior takes
    and which the conflicting context states."""
    return Phase1Pair(
        pair_id=cand.candidate_id,
        domain=domain if domain is not None else cand.domain,
        template=cand.template,
        question=cand.question,
        context_supportive=cand.context_for(prior_value),
        context_conflicting=cand.context_for(context_value),
        answer_context_token=context_value,
        answer_prior_token=prior_value,
        measured_prior_logit_diff=measured_prior_logit_diff,
        is_control=is_control,
    )


def phase1_pairs_path(model_name: str) -> Path:
    return REPO_ROOT / "data" / "pairs" / f"phase1_{model_name}.jsonl"


def phase5_pairs_path(model_name: str) -> Path:
    return REPO_ROOT / "data" / "pairs" / f"phase5_{model_name}.jsonl"


def load_phase5_pairs(model_name: str) -> list[Phase1Pair]:
    pairs = []
    with phase5_pairs_path(model_name).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(Phase1Pair(**json.loads(line)))
    return pairs


def save_phase1_pairs(pairs: list[Phase1Pair], model_name: str) -> Path:
    path = phase1_pairs_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
    return path


def load_phase1_pairs(model_name: str) -> list[Phase1Pair]:
    pairs = []
    with phase1_pairs_path(model_name).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(Phase1Pair(**json.loads(line)))
    return pairs


def build_prompt(context: str | None, question: str) -> str:
    """Render a context (or None for the no-context/neutral probe) + question
    into the prompt whose next token is read out as the answer logit."""
    if context is None:
        return f"Question: {question}\nAnswer:"
    return f"Context: {context}\n\nQuestion: {question}\nAnswer:"


def validate_pairs(pairs: list[Pair], models: dict[str, object]) -> None:
    """Assert every parametric_answer/edited_answer is a single token (with a
    leading space) in every given model's vocabulary. Raises ValueError
    listing ALL failures at once rather than failing on the first one, since
    silently measuring a multi-token answer's first sub-token would be a
    silent methodological error, not a crash we can ignore."""
    failures: list[str] = []
    for pair in pairs:
        for field in ("parametric_answer", "edited_answer"):
            answer = getattr(pair, field)
            for model_name, model in models.items():
                try:
                    model.to_single_token(" " + answer)
                except AssertionError:
                    str_tokens = model.to_str_tokens(" " + answer, prepend_bos=False)
                    failures.append(
                        f"{pair.pair_id}.{field}={answer!r} is not a single token in "
                        f"{model_name} (splits as {str_tokens})"
                    )
    if failures:
        raise ValueError(
            f"{len(failures)} tokenization failures found:\n" + "\n".join(failures)
        )


def save_pairs_jsonl(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")


def load_pairs_jsonl(path: Path) -> list[Pair]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(Pair(**json.loads(line)))
    return pairs


def main() -> None:
    import torch
    from transformer_lens import HookedTransformer

    torch.manual_seed(0)
    print("Loading gpt2 and pythia-160m to validate answer tokenization...")
    models = {
        "gpt2": HookedTransformer.from_pretrained("gpt2", device="cpu"),
        "pythia-160m": HookedTransformer.from_pretrained("pythia-160m", device="cpu"),
    }

    ids = [p.pair_id for p in PAIRS]
    assert len(ids) == len(set(ids)), "duplicate pair_id found"

    validate_pairs(PAIRS, models)
    print(f"All {len(PAIRS)} pairs validated: every answer is a single token in both models.")

    save_pairs_jsonl(PAIRS, PAIRS_PATH)
    print(f"Wrote {len(PAIRS)} pairs to {PAIRS_PATH}")


if __name__ == "__main__":
    main()
