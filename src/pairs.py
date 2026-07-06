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


def phase1_pairs_path(model_name: str) -> Path:
    return REPO_ROOT / "data" / "pairs" / f"phase1_{model_name}.jsonl"


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
