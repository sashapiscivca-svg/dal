# Responsible use

[Українська](RESPONSIBLE_USE.uk.md)

dal exists to help **find and keep sight of people who need help**: people swept away by floods and tsunamis, lost in forests and mountains, or stranded after earthquakes. Once a rescuer or a person detector has pointed at a likely person in drone video, dal keeps a box on that person in the following frames, so that the find can be confirmed and a team can be sent. We built it to run on very cheap hardware so that volunteer and under-funded rescue teams can use it too. We measure success in lives saved.

## What the licence says, and what we ask

The code and the released weights are under the **MIT License**. It places no legal restriction on fields of use, and we will not pretend that it does. What follows is our **public request and commitment**, not a licence condition.

### We ask you not to use dal for

1. **Military purposes**: reconnaissance, detecting, tracking or targeting people or equipment for combat, fire control, target designation, or any system meant to inflict harm. This includes "defensive" and ISR uses.
2. **Persecution**: finding, tracking or pressuring people for their beliefs, religion, origin, language, sex, orientation, political or civic activity, or for taking part in assemblies and protests.
3. **Mass or covert surveillance**: continuous monitoring of populations, or following individuals without their knowledge and a lawful basis.
4. **Repression**: helping any actor suppress rights and freedoms, detain arbitrarily, displace people by force, or violate international humanitarian or human-rights law.
5. **Acting against people who need protection**: locating refugees or migrants in order to push them back or detain them unlawfully, or hunting people in distress instead of helping them.
6. **Identifying individuals**: face recognition, biometrics, linking a track to an identity, re-identifying a person across flights, days or places, or inferring group membership from appearance.
7. **Any use intended or foreseeably likely to cause death, injury, coercion or intimidation.**

If you cannot tell whether a use helps or harms, please treat it as out of scope.

### Uses we built it for

- Search and rescue of people; searching for missing persons at the request of rescue services or families.
- Disaster response: floods, tsunamis, earthquakes, wildfires, avalanches.
- Research and teaching consistent with the list above.
- Humanitarian missions overseen by independent organisations.

## How the design supports this

- **A box in the image, nothing more.** dal outputs where the target is in the current frame (a box, a confidence and a "lost" flag). It has no identity, face, age, sex or other attribute outputs, and we will not add them.
- **Tracking only for the duration of an event.** A tracker object holds its state only while it runs. Nothing about a target is saved between runs, and dal has no way to recognise the same person in another video, flight or day. We will not add cross-session re-identification.
- **It does not steer anything.** dal contains no flight control, no gimbal or camera control, no autopilot or MAVLink interface, no geolocation and no "follow the target" logic. It reports image coordinates to a human or to software that a human supervises. Do not wire its output to automated actions against people.
- **Runs locally.** Everything runs on the device. Nothing is sent to us or to any cloud.
- **A human decides.** dal suggests; a rescuer decides.
- **Honest limits.** We publish how often the tracker loses people, how much it drifts, and where it fails (see the [README](README.md#what-is-not-proven) and [MODEL_CARD.md](MODEL_CARD.md)). A lost person can mean a death, so we do not hide weaknesses.
- **Clean provenance.** Only civilian, openly licensed data; no third-party pretrained weights; no datasets of military origin or funding.

## Operational safety for rescue teams

- **A lost track does not mean the person is gone.** On thermal footage the tracker loses about one person in three within 100 frames without detector corrections. Keep the search going, re-run the detector, and look again.
- **A held track can still be wrong.** Most losses we measured on thermal footage were "confidently wrong": the tracker kept reporting a box that had drifted onto the background. Confirm a find visually before dispatching a team.
- Validate the tracker on your own camera, altitude and terrain before an operation, and keep the human visual search running.
- Follow aviation rules, your organisation's SAR protocols, and local privacy law on recording and storing footage of people. Delete footage that the mission no longer needs.
- Fit a heatsink to a Raspberry Pi Zero 2 W used in flight. It throttles above about 80 °C.

## What we do

- We provide no support, consulting, hosting or partnership to projects that conflict with this document, and we will publicly disclose credible reports of such use.
- Changes that strengthen this document may be made at any time. Any weakening will be justified publicly and will take effect no sooner than 30 days after it is announced.

## Reporting

To report misuse, or a safety problem (for example a systematic failure mode that could cost lives), follow [SECURITY.md](SECURITY.md). We read every report.
