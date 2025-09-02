import asyncio
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

NARRATIVE_DB = {
    "RUN123": "Patient was transported to the hospital with minor injuries.",
    "RUN456": "Patient required intubation during transport. Patient was also given an IV.",
    "RUN789": "Patient was treated on scene and refused transport.",
    "RUN000": "EMS crew arrived at a camp ground. Pt was found crying with broken arm. Pt was splinted "
              "and transported to hospital."
}

class DatabaseConn:
    async def get_narrative(self, run_number: str) -> str:
        return NARRATIVE_DB.get(run_number, "No narrative found.")

@dataclass
class BillerDependencies:
    run_number: str
    db: DatabaseConn

class BillingOutput(BaseModel):
    level_of_service: Literal['BLS', 'ALS', 'ALS-2', 'Transport Not Taken'] = Field(description="The level of service provided by EMS crew")
    rationale: str = Field(description="The rationale for the chosen level of service")


medical_biller = Agent(
    "openai:gpt-5-nano",
    deps_type=BillerDependencies,
    output_type=BillingOutput,
    system_prompt=(
        "You are a certified medical biller specializing in EMS billing. If giving rationale, be brief and to the point."
        " Rationale should start with 'Pt was ...'"
    )
)

@medical_biller.tool
async def fetch_narrative(ctx: RunContext[BillerDependencies]) -> str:
    return await ctx.deps.db.get_narrative(ctx.deps.run_number)

async def main():
    db = DatabaseConn()
    for run in ["RUN000", "RUN123", "RUN456", "RUN789"]:
        deps = BillerDependencies(run_number=run, db=db)
        result = await medical_biller.run("Determine the appropriate level of service based on the"
                                          " ambulance run's narrative. Give rationale before providing "
                                          "final answer.", deps=deps)
        print(f"Run: {run}, Level of Service: {result.output.level_of_service}, Rationale: {result.output.rationale}")

"""
Example Output:
Run: RUN000, Level of Service: BLS, Rationale: Pt was found with a broken arm, splinted, and transported to hospital; no ALS interventions were required.
Run: RUN123, Level of Service: BLS, Rationale: Pt was stable on assessment; transported to the hospital with minor injuries; no ALS interventions required; BLS transport appropriate.
Run: RUN456, Level of Service: ALS, Rationale: Pt was intubated during transport and IV access was obtained, indicating advanced airway management and IV therapy, which aligns with ALS-level EMS service.
Run: RUN789, Level of Service: TNT, Rationale: Pt was treated on scene and refused transport; therefore transport not taken (TNT) is appropriate.
"""

if __name__ == "__main__":
    asyncio.run(main())