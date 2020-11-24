def objective(model):
    obj = sum(pi_3[surgery.get_surgery_id()].value() * model.omega1[surgery] for surgery in planned_surgery)
    obj += sum(pi_4[surgery1.get_surgery_id(), surgery2.get_surgery_id(), r].value() * model.omega2[surgery1, surgery2, r] for r in R for surgery1 in dict_surgery_room[r] for surgery2 in dict_surgery_room[r] if surgery1 != surgery2)
    obj += sum(pi_5[surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id()].value() * model.omega3[surgery1, surgeon] for surgeon in planned_surgeon for surgery1 in dict_surgery_surgeon[surgeon] for surgery2 in dict_surgery_surgeon[surgeon] if surgery1 != surgery2)
    obj += sum(pi_7[surgery.get_surgery_id(), r].value() * model.omega4[surgery, r] for r in R for surgery in dict_surgery_room[r])
    obj += sum(pi_8[surgery.get_surgery_id(), surgeon.get_surgeon_id()].value() * model.omega5[surgery, surgeon] for surgeon in planned_surgeon for surgery in dict_surgery_surgeon[surgeon])
    obj -= sum(pi_11[surgeon.get_surgeon_id()].value() * model.omega6[surgeon] for surgeon in planned_surgeon)
    return obj

def rule1(model):
    return sum(pyo.log(aprox_norm_cdf((pyo.log(model.omega1[surgery]) - distribution_dict[surgery.get_group(), 'preparation_mu']) / np.sqrt(distribution_dict[surgery.get_group(), 'preparation_sigma']))) for surgery in planned_surgery) == np.log(alpha)

def rule2(model):
    if len(model.comb_S) == 0:
        return pyo.Constraint.NoConstraint
    else:
        mu = {}
        sigma = {}
        for r in R:
            for surgery1 in dict_surgery_room[r]:
                for surgery2 in dict_surgery_room[r]:
                    if surgery1 != surgery2:
                        mean = distribution_dict[surgery1.get_group(), 'preparation_mean'] + distribution_dict[surgery1.get_group(), 'surgery_mean'] + distribution_dict[surgery2.get_group(), 'cleaning_mean']
                        variance = distribution_dict[surgery1.get_group(), 'preparation_variance'] + distribution_dict[surgery1.get_group(), 'surgery_variance'] + distribution_dict[surgery2.get_group(), 'cleaning_variance']
                        sigma[surgery1, surgery2] = np.log(variance / mean ** 2 + 1)
                        mu[surgery1, surgery2] = np.log(mean) - sigma[surgery1, surgery2] / 2
        return sum(pyo.log(aprox_norm_cdf((pyo.log(model.omega2[surgery1, surgery2, r]) - mu[surgery1, surgery2]) / np.sqrt(sigma[surgery1, surgery2]))) for (surgery1, surgery2, r) in model.comb_S) == np.log(alpha)

def rule3(model):
    if len(comb_S2) == 0:
        return pyo.Constraint.NoConstraint
    else:
        return sum(pyo.log(aprox_norm_cdf((pyo.log(model.omega3[surgery1, surgeon]) - distribution_dict[surgery1.get_group(), 'surgery_mu']) / np.sqrt(distribution_dict[surgery1.get_group(), 'surgery_sigma']))) for (surgery1, surgeon) in model.comb_S2) == np.log(alpha)

def rule4(model):
    if len(comb_S3) == 0:
        return pyo.Constraint.NoConstraint
    else:
        mu = {}
        sigma = {}
        for (surgery, r) in model.comb_S3:
            mean = distribution_dict[surgery.get_group(), 'surgery_mean'] + distribution_dict[surgery.get_group(), 'cleaning_mean']
            variance = distribution_dict[surgery.get_group(), 'surgery_variance'] + distribution_dict[surgery.get_group(), 'cleaning_variance']
            sigma[surgery] = np.log(variance / mean ** 2 + 1)
            mu[surgery] = np.log(mean) - sigma[surgery] / 2
        return sum(pyo.log(aprox_norm_cdf((pyo.log(model.omega4[surgery, r]) - mu[surgery]) / np.sqrt(sigma[surgery]))) for (surgery, r) in model.comb_S3) == np.log(alpha)

def rule5(model):
    if len(comb_S4) == 0:
        return pyo.Constraint.NoConstraint
    else:
        return sum(pyo.log(aprox_norm_cdf((pyo.log(model.omega5[surgery, surgeon]) - distribution_dict[surgery.get_group(), 'surgery_mu']) / np.sqrt(distribution_dict[surgery.get_group(), 'surgery_sigma']))) for (surgery, surgeon) in model.comb_S4) == np.log(alpha)

def rule6(model):
    mu = {}
    sigma = {}
    for surgeon in planned_surgeon:
        mean = sum(distribution_dict[surgery.get_group(), 'surgery_mean'] for surgery in dict_surgery_surgeon[surgeon])
        variance = sum(distribution_dict[surgery.get_group(), 'surgery_variance'] for surgery in dict_surgery_surgeon[surgeon])
        sigma[surgeon] = np.log(variance / mean ** 2 + 1)
        mu[surgeon] = np.log(mean) - sigma[surgeon] / 2
    return sum(pyo.log(1 - aprox_norm_cdf(pyo.log(model.omega6[surgeon]) -  mu[surgeon]) / np.sqrt(sigma[surgeon])) for surgeon in model.planned_surgeon) == np.log(alpha)
comb_S = []
for r in R:
    surgery_in_r = dict_surgery_room[r]
    for surgery1 in surgery_in_r:
        for surgery2 in surgery_in_r:
            if surgery1 != surgery2:
                comb_S.append((surgery1, surgery2, r))

comb_S2 = []
for surgeon in planned_surgeon:
    surgery_by_k = dict_surgery_surgeon[surgeon]
    for surgery1 in surgery_by_k:
        for surgery2 in surgery_by_k:
            if surgery1 != surgery2:
                comb_S2.append([surgery1, surgeon])

comb_S3 = []
for r in R:
    for surgery in dict_surgery_room[r]:
        comb_S3.append([surgery, r])
comb_S4 = []
for surgeon in planned_surgeon:
    for surgery in dict_surgery_surgeon[surgeon]:
        comb_S4.append([surgery, surgeon])
aux_model = pyo.ConcreteModel()
aux_model.R = pyo.Set(initialize=R)
aux_model.planned_surgeon = pyo.Set(initialize=planned_surgeon)
aux_model.comb_S = pyo.Set(initialize=comb_S)
aux_model.comb_S2 = pyo.Set(initialize=comb_S2)
aux_model.comb_S3 = pyo.Set(initialize=comb_S3)
aux_model.comb_S4 = pyo.Set(initialize=comb_S4)
aux_model.omega1 = pyo.Var(planned_surgery, domain=pyo.Reals)
aux_model.omega2 = pyo.Var(comb_S, domain=pyo.Reals)
aux_model.omega3 = pyo.Var(comb_S2, domain=pyo.Reals)
aux_model.omega4 = pyo.Var(comb_S3, domain=pyo.Reals)
aux_model.omega5 = pyo.Var(comb_S4, domain=pyo.Reals)
aux_model.omega6 = pyo.Var(planned_surgeon, domain=pyo.Reals)

aux_model.objective = pyo.Objective(rule=objective)
aux_model.constraint1 = pyo.Constraint(rule=rule1)
aux_model.constraint2 = pyo.Constraint(rule=rule2)
aux_model.constraint3 = pyo.Constraint(rule=rule3)
aux_model.constraint4 = pyo.Constraint(rule=rule4)
aux_model.constraint5 = pyo.Constraint(rule=rule5)
aux_model.constraint6 = pyo.Constraint(rule=rule6)
res = pyo.SolverFactory("scip").solve(aux_model, tee=True)
objective_value = aux_model.OBJ()