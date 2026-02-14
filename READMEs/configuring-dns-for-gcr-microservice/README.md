# TLDR

Documenting steps of getting the `kalygo-completion-api` microservice hosted on GCP with DNS provided by AWS Route 53

## STEPS

- Deployed a new GCR service
- Verified the `kalygo.io` domain with Google Search Console
  - Needed to add a TXT record to Route53
- Created a GCR "Domain Mapping"
  - From `completion.kalygo.io` to the `kalygo-completion-api-service`
  - console.cloud.google.com/run/domains
  - After "Domain Mapping" is created will need to create a DNS record in Route53 or DNS config
    - Name: completion
    - Type: CNAME
    - Data: ghs.googlehosted.com.
- After 15 minutes the DNS & SSL config seemed to be set up
- For testing
  - `curl -v 'https://completion.kalygo.io/'`
