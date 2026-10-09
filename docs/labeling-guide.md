# Labeling Guide

How to assign one of the 16 classifier categories to an owner-language symptom
description. Used for new training rows (human-written or LLM-generated) and for
reviewing flagged labels. The categories are ICD-11-style chapters consolidated by
`data/label_map.json`; the rules below follow how the majority of existing rows
are labeled, so new data does not contradict old data.

Label what the **text** supports, not the most likely diagnosis behind it. If the
text alone cannot support any single category (e.g. "routine booster, no
concerns"), the row is unlabelable and must not be used.

## Categories

| Category | Typical owner-visible signs | Raw labels folded in |
|---|---|---|
| Blood Disorders | pale or white gums, weakness with paleness, unexplained bruising, nosebleeds, bleeding that won't stop, dark/red urine with paleness | anemia, blood-forming organs |
| Cardiovascular Conditions | tiring fast on walks, fainting/collapse with exertion, cough worse at night or lying down with fatigue, swollen belly from fluid, fast breathing at rest in an older pet, blue-tinged gums, heartworm | heart disease, heartworm |
| Digestive Issues | vomiting, diarrhea, constipation, straining to poop, gas/bloating, not eating, drooling with mouth pain, bad teeth, gum disease, swallowed object stuck | gastroenteritis, colitis, GI blockage |
| Ear Conditions | head shaking, scratching ears, ear smell, ear discharge/wax, red ear canal, swollen ear flap, head tilt with ear pain, ear mites with ear signs | ear infections |
| Eye Conditions | squinting, eye discharge, red eye, cloudy eye, rubbing eye, third eyelid showing, bulging eye, sudden vision loss | visual system |
| Genitourinary Conditions | straining to pee, peeing small amounts often, blood in urine, accidents in the house, licking genitals, crying in the litter box, kidney decline (thirst + vomiting + weight loss + bad breath in an old cat) | cystitis, UTI, bladder stones, chronic kidney disease |
| Immune System Disorders | allergic reaction: hives, sudden face/muzzle swelling, itching tied to a food/season with GI or ear flare, autoimmune signs across several body systems | allergies, immune system |
| Infectious and Parasitic Diseases | fleas, ticks, worms in poop or vomit, scooting with worms, mange/skin mites, ringworm, parvo-like bloody diarrhea in an unvaccinated puppy, cat flu after shelter/cattery exposure, fever with lethargy after contact with sick animals | parasites, fleas, parvovirus, feline herpesvirus |
| Injury and Poisoning | ate chocolate/grapes/xylitol/lily/rat bait/medication, hit by a car, fall, bite wound, cut paw, torn nail, burn, bee sting site, choking | injury, poisoning, trauma, puncture wounds |
| Metabolic and Endocrine Disorders | drinking and peeing much more **plus** big appetite or weight loss (diabetes, hyperthyroid), pot belly with thinning coat and panting (Cushing's), weight gain with lethargy and cold-seeking (hypothyroid), sweet-smelling breath | diabetes, hyperthyroidism, endocrine/nutritional |
| Musculoskeletal Conditions | limping, stiffness after rest, reluctance to jump or climb stairs, bunny-hopping, swollen joint, yelping when picked up with back pain, hind-leg weakness from the back/disc | osteoarthritis, hip dysplasia, intervertebral disc disease |
| Neoplasms | a lump or mass that is growing, changing, or bleeding, lump on mammary chain, mouth mass, unexplained weight loss alongside a mass | tumours, neoplasms |
| Neurological and Behavioural Disorders | seizures/fits, trembling episodes, circling, head pressing, wobbly gait without pain, sudden blindness with confusion, disorientation in old age, anxiety, aggression, compulsive licking/tail chasing, inappropriate marking from stress | nervous system, behavioural, epilepsy, cognitive dysfunction |
| Reproductive Conditions | pregnancy, labor and delivery problems, heat cycle concerns, false pregnancy, newborn puppy/kitten problems, congenital defects present from birth, neutering-wound/post-castration concerns | pregnancy, perinatal, developmental anomalies, orchiectomy |
| Respiratory Conditions | sneezing, nasal discharge, honking/hacking cough, kennel cough, noisy or labored breathing, snoring/snorting in flat-faced breeds, reverse sneezing, wheezing | respiratory, kennel cough, URI, pneumonia, brachycephalic syndrome |
| Skin Conditions | itching/scratching skin without parasites mentioned, hot spots, rashes, hair loss patches, scabs, dandruff, chewing paws, overgrooming, red belly skin, atopic skin flares | dermatitis, atopy, pruritus, psychogenic alopecia |

## Boundary rules for the confusable pairs

- **Skin vs Infectious/Parasitic:** a named parasite or fungus (fleas, flea dirt,
  ticks, mange, ringworm) → Infectious/Parasitic. Itching, hair loss or rash with
  no parasite mentioned → Skin.
- **Ear vs Infectious/Parasitic:** ear mites or yeast with ear signs only → Ear.
- **Skin vs Immune:** chronic itchy skin, paw chewing, seasonal skin flares → Skin.
  Acute allergic reaction (hives, face swelling) or allergy affecting several
  systems → Immune.
- **Cardiovascular vs Respiratory:** cough plus exercise intolerance, fainting,
  fluid belly or fast resting breathing in an older pet → Cardiovascular. Cough or
  breathing noise with sneezing, nasal discharge, honking, or kennel exposure →
  Respiratory.
- **Respiratory vs Infectious:** sneezing/nasal discharge alone → Respiratory. The
  same signs with an explicit contagion context (shelter, cattery, littermates
  sick, unvaccinated) → Infectious.
- **Metabolic vs Genitourinary:** more thirst and urine with big appetite, weight
  loss despite eating, pot belly, or coat changes → Metabolic. Straining, blood,
  small frequent amounts, accidents, or old-cat kidney decline → Genitourinary.
- **Neurological vs Musculoskeletal:** pain-driven movement problems (limp,
  stiffness, back pain, disc) → Musculoskeletal. Seizures, tremor episodes,
  circling, head tilt without ear signs, confusion, behaviour → Neurological.
- **Blood vs Immune:** paleness, bruising, bleeding → Blood, even when the
  underlying cause is immune-mediated.
- **Digestive vs Infectious:** vomiting/diarrhea alone → Digestive. Visible worms,
  or bloody diarrhea in an unvaccinated puppy → Infectious.
- **Digestive vs Injury:** swallowed a toxin → Injury and Poisoning. Swallowed a
  non-toxic object now vomiting/not pooping → Digestive.

## Writing rules for new rows

- First-person owner voice, lay vocabulary, 6–60 words. Typos, hedging, and
  emotional phrasing are fine; vet jargon and lab values are not.
- Never name the category or a diagnosis ("I think it's diabetes", "skin
  condition"). Describe what the owner sees. `ml_pipeline/synthetic_quality.py`
  rejects rows containing the category label.
- Vary species (mostly dogs and cats, some rabbits and other small pets), age,
  breed, duration, and how many signs are mentioned.
- Generated rows use their own `record_type` (e.g. `Synthetic Owner (Claude)`),
  never `Owner Observation`. `prepare_dataset.py` carves the owner holdout from
  `Owner Observation`, so mislabeled provenance would leak generated text into the
  evaluation set.
