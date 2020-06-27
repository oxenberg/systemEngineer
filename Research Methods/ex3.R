
world_data = read_csv('../Desktop/Masters/courses/Research Methods/ex3/World95/World95-updated.csv')

world_data<- dplyr::select(world_data, gdp_cap, populatn, density, urban, calories, lit_male, lit_fema)%>%na.omit()

attach(world_data)
m1 = lm(world_data$gdp_cap ~ world_data$populatn + world_data$density + world_data$urban)
summary(m1)
m2 = lm(world_data$gdp_cap ~ world_data$populatn + world_data$density + world_data$urban + world_data$calories)
summary(m2)
m3 = lm(world_data$gdp_cap ~ world_data$populatn + world_data$density + world_data$urban + world_data$calories +world_data$lit_male + world_data$lit_fema)
summary(m3)
anova(m2,m3)

world_data<- dplyr::select(world_data, -country)%>%na.omit()
resForward = stepAIC(lm(gdp_cap~1, world_data), scope=~populatn+density+urban+lifeexpf+lifeexpm+literacy+pop_incr+babymort+calories+aids+birth_rt+death_rt+aids_rt+lg_aidsr+b_to_d+fertilty+log_pop+lit_fema+lit_male, direction = "forward")

attach(world_data)
m1 = lm(world_data$gdp_cap ~ 1)
summary(m1)

m2 = lm(world_data$gdp_cap ~  world_data$calories)
summary(m2)
anova(m1,m2)

m3 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids)
summary(m3)
anova(m2,m3)

m4 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density )
summary(m4)
anova(m3,m4)

m5 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr)
summary(m5)
anova(m4,m5)

m6 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm)
summary(m6)
anova(m5,m6)

m7 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm + world_data$death_rt)
summary(m7)
anova(m6,m7)

m8 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm + world_data$death_rt +world_data$lit_fema)
summary(m8)
anova(m7,m8)

m9 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm + world_data$death_rt +world_data$lit_fema + world_data$urban)
summary(m9)
anova(m8,m9)

m10 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm + world_data$death_rt +world_data$lit_fema + world_data$urban + world_data$birth_rt)
summary(m10)
anova(m9,m10)

m11 = lm(world_data$gdp_cap ~ world_data$calories + world_data$aids + world_data$density + world_data$lg_aidsr + world_data$lifeexpm + world_data$death_rt +world_data$lit_fema + world_data$urban + world_data$birth_rt + world_data$fertilty)
summary(m11)
anova(m10,m11)

