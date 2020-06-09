search_data = read_csv('../Desktop/Masters/courses/Research Methods/ex2/Ex2Data/Search.csv')

search_data = search_data%>% select(ProblemTypeId, DistanceFromGoal, RunTime)
search_data%>%
  group_by(ProblemTypeId)%>%
  summarise(mean_distance = mean(DistanceFromGoal), 
            mean_runtime = mean(RunTime),
            std_distance = sd(DistanceFromGoal),
            std_runtime = sd(RunTime))

###########33
attach(search_data)

by(search_data, ProblemTypeId, summary)
by(search_data, search_data$ProblemTypeId, var)

#########
bartlett.test(DistanceFromGoal, ProblemTypeId)

anovaModel <- aov(DistanceFromGoal ~ ProblemTypeId) 
summary(anovaModel)
#############
bartlett.test(RunTime, ProblemTypeId)

oneway.test(RunTime~ProblemTypeId, var.equal=FALSE)

anovaModel <- aov(RunTime ~ as.factor(ProblemTypeId), search_data) 
summary(anovaModel)
TukeyHSD(anovaModel)



#############

exp = read_csv('../Desktop/Masters/courses/Research Methods/ex2/Ex2Data/ExpRec.csv')
exp = exp%>%select(ageGroup,recSystemTypeId,expType, score)

attach(exp)

by(exp, ageGroup, summary)
by(exp, ageGroup, var)

by(exp, recSystemTypeId, summary)
by(exp, recSystemTypeId, var)

by(exp, expType, summary)
by(exp, expType, var)


### 
bartlett.test(score, ageGroup)


bartlett.test(score, recSystemTypeId)


bartlett.test(score, expType)
exp$recSystemTypeId<-factor(exp$recSystemTypeId)
exp$ageGroup<-factor(exp$ageGroup)
exp$expType<-factor(exp$expType)
anovaModel <-aov(score ~ ageGroup + recSystemTypeId + expType + ageGroup*recSystemTypeId*expType, exp)

summary(anovaModel)

TukeyHSD(anovaModel,'recSystemTypeId')

interaction.plot(ageGroup, expType,  score)

