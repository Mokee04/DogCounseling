<!-- # 파생변수 계산 예시 -->
<!-- ```
info['current_date'] = datetime.now().strftime('%Y-%m-%d')
info['pet_age'] = round((pd.to_datetime(info['current_date']) - pd.to_datetime(info['pet_birth_date'])).days/365, 1)
info['years_current-adopted'] = round((pd.to_datetime(info['current_date']) - pd.to_datetime(info['pet_adopted_at'])).days/365, 1)
info['weeks_adopted-birth'] = round((pd.to_datetime(info['pet_adopted_at']) - pd.to_datetime(info['pet_birth_date'])).days/7, 1)
``` -->

### Dog Care Counseling: Pre-Questionnaire Summary

- **Report Date**: {{current_date}}
- **Note**: The dog participating in this counseling session will be referred to as the 'Protagonist Dog'.

---

#### 1. Protagonist Dog's Information

##### Basic Profile
- **Name**: {{pet_name}}
- **Birth Date**: {{pet_birth_date}} (Age: {{pet_age}} years)
- **Adoption Date**: {{pet_adopted_at}}
  - Weeks between birth and adoption: {{weeks_adopted-birth}} weeks
  - Years living together: {{years_current-adopted}} years
- **Breed**: {{pet_breed}}
- **Sex**: {{pet_gender}} (Neutered?: {{pet_is_neutered}})
- **Adoption Method**: {{env2}}

##### Health Record
- **Diseases (last 6 months)**: {{med_disease}}
- **Allergies**: {{med_allergy}}

##### Temperament & Personality
- **DCSI Personality Type**: {{dcsi_label}} ({{dcsi_code}})
- **Preferred Rewards**: {{pref_rewards}}
- **Triggers for Alertness/Discomfort**: {{fear_triggers}}
- **Body Parts Sensitive to Touch/Contact**: {{fear_touch}}
    
##### Social & Environmental Experiences
- **Social Interactions during Socialization Period (people met)**: {{social_interactions}}
- **Known Cues (in a calm environment, >=70% success rate)**: {{social_trains}}
- **Reaction to Strangers Visiting Home**: {{fear_stranger}}
- **Reaction to Vets/Nurses at the Clinic**: {{fear_clinic}}
- **Can the dog play well with other dogs?**: {{fear_dog}}
- **Does the dog follow the guardian wherever they go?**: {{fear_separation}}

##### Behavioral Patterns
- **Resource Guarding Situations**: {{behavior_guarding}}
- **Behaviors When Left Alone**: {{behavior_separation}}
- **Demand-Seeking Behaviors (when guardian is busy)**: {{behavior_demand}}
- **Impulsivity (reaction to curious/fun stimuli)**: {{behavior_impulsive}}
- **Over-arousal (excessive physical reaction when excited)**: {{behavior_overarousal}}
- **Compulsive Licking (to the point of skin damage)**: {{behavior_licking}}
- **Reaction to Denial (intensified demands when refused)**: {{behavior_denial}}

##### Aggression History
- **Situations Showing Threatening or Aggressive Behavior**:
  - {{behavior_aggression}}
- **Bite History & Intensity (if any)**:
  - {{behavior_bite}}

---

#### 2. User(Guardian) Information

- **Caregiving Role for the protagonist dog**: {{env1}}
- **Interests for Counseling/Behavior Correction**: {{counseling_topics}}

---

#### 3. Living Environment

- **Living Space Type**: {{env3}} (Tied-up at specific area?: {{env5}})
- **Number of Bedrooms**: {{env4}}
- **Dog's Personal Shelter Type**: {{life_space}}

##### Family Composition
- **Number of Human Family Members**: {{env7}}
- **Presence of Young or Elderly Members**: {{env8}}
- **Number of Total Animals(including Protagonist Dog)**: {{env9}}

---

#### 4. Protagonist Dog's Lifestyle

- **Monthly Pet Care Cost (KRW)**: {{env6}}(만원)
- **Walking Routine**:
  - **Frequency**: {{life_walk_day}} times/day, {{life_walk_week}} times/week
  - **Duration**: {{life_walk_hour}} minutes/walk
- **Other Activities (type, frequency, duration)**: {{life_play}}
- **Time Spent with Human Family per Day**: {{life_human}} hours
- **Time Spent Alone per Day**: {{life_alone}} hours

---