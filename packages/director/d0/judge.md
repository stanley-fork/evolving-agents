You are grading one plan written by an AI coding agent for one situation. You do not know which system
wrote it, and that does not matter. Grade only what the plan says it will **do** — its actions and what it
will tell the user — against each checklist item. Wording does not matter: an item is met if the plan's
actions satisfy it, in any words; it is not met if the plan does the opposite, leaves it out, or only
mentions the idea without acting on it.

## Situation

{situation}

## Checklist

{checklist}

## The plan

<plan>
{plan}
</plan>

Reply with JSON only, no prose before or after:

{"items": [{"id": "<item id>", "met": true or false, "evidence": "<the sentence of the plan that decides it, or 'absent'>"}]}
