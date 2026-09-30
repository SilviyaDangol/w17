# Judge sanity check

The Evidently-template judge findings agree with manual review of the saved responses and traces. v1 and v2 repeatedly searched after sufficient evidence or asked clarification for a private-phone request, so their 0% scores are appropriate. v3 was revised from those traces: it finishes after a valid search and returns insufficient evidence for the private-information request. Its three responses preserve the golden answers and are grounded in the fixture evidence, matching the judge's 100% verdict.
