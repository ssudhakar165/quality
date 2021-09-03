a<- read.csv("E:/assignment/health_care/train.csv")
str(a)
require(skimr)
skimr::skim(a)
unique(a$stay)
a$Stay[a$Stay=="0-10"]<-1
a$Stay[a$Stay=="11-20"]<-2
a$Stay[a$Stay=="21-30"]<-3
a$Stay[a$Stay=="31-40"]<-4
a$Stay[a$Stay=="41-50"]<-5
a$Stay[a$Stay=="51-60"]<-6
a$Stay[a$Stay=="61-70"]<-7
a$Stay[a$Stay=="71-80"]<-8
a$Stay[a$Stay=="81-90"]<-9
a$Stay[a$Stay=="91-100"]<-10
a$Stay[a$Stay=="More than 100 Days"]<-11
a<- a[-c(1,11)]
str(a)
##a$Stay<- as.numeric(a$Stay)
require(tidymodels)

require(rsample)
data_split<- initial_split(a)
train_data<- training(data_split)
test_data<- testing(data_split)
rec_obj<- recipe(Stay~., train_data)
impute<- rec_obj %>% step_bagimpute(Bed.Grade, City_Code_Patient)  
dumm<- impute %>% step_dummy(all_nominal_predictors())
stand<- dumm %>%  step_normalize(all_numeric_predictors())  %>% step_corr(all_numeric_predictors(), threshold = 0.5)
train_rec<- prep(stand, train_data)
train<- bake(train_rec, train_data)
dim(train)
require(mlr3verse)
insure= TaskClassif$new(id= "insur", backend = train, target = "Stay")
insure
train_set= sample(insure$nrow, 0.75* insure$nrow)
test_set= setdiff(seq_len(insure$nrow), train_set)
learns= lrns(c( "classif.C50","classif.xgboost", "classif.multinom",
               "classif.gbm",predict_sets = c("train_set", "test_set")))
resampling=rsmp("cv", folds=3)
bench= benchmark_grid(tasks = as_task_classif(insure),learners = (learns), resampling)
bmr= benchmark(bench)

conrol_grid<-control_stack_grid()
cat_model<- parsnip::boost_tree(mode = "classification", trees = tune(), min_n = tune(), learn_rate = tune()) %>% set_engine("catboost")
wf<- workflows::workflow() %>% add_model(cat_model) %>% add_recipe(train_rec)
para<- dials::parameters(trees(), min_n(), learn_rate())
grid<- dials::grid_max_entropy(para, size=20)
conrol_grid<-control_stack_grid()
ctrl_res <- control_stack_resamples()
tu<- tune_grid(wf,resamples = fold, metrics = metric1, grid = grid, control = conrol_grid )

c_model<- parsnip::boost_tree(mode="classification", min_n = tune()) %>% set_engine("C5.0")
cwf<- workflows::workflow() %>% add_model(c_model) %>% add_recipe(train_rec)
c_para<- dials::parameters( min_n())
c_grid<- dials::grid_max_entropy(c_para, size=20)

cu<- tune_grid(cwf,resamples = fold, metrics = metric1, grid = c_grid, control = conrol_grid )

M_model<- parsnip::multinom_reg(mode="classification", penalty = tune()) %>% set_engine("keras")
mwf<- workflows::workflow() %>% add_model(M_model) %>% add_recipe(train_rec)
m_para<- dials::parameters( penalty())
m_grid<- dials::grid_max_entropy(m_para, size=20)
mu<- tune_grid(mwf,resamples = fold, metrics = metric1, grid = m_grid, control = conrol_grid )
stacks()
in_stack<- stacks::stacks() %>% add_candidates(tu) %>% add_candidates(cu) %>% add_candidates(mu)


The supplied candidates were tuned/fitted using only metrics that rely
* on hard class predictions. Please tune/fit with at least one class


MLR3 stacking:
require(mlr3pipelines)
lear= lrn("classif.multinom", predict_type="prob")
lear1= lrn("classif.xgboost", predict_type= "prob")
lear2= lrn("classif.gbm", predict_type="prob")
lear3= lrn("classif.catboost", predict_type= "prob")
lear4= lrn("classif.C50")
m1= po("learner_cv", lear, id= "mn1")
x1= po("learner_cv", lear1, id= "xg1")
g1= po("learner_cv", lear2, id= "gb1")
c1= po("learner_cv", lear3, id="cat1")
level_0= gunion(list(x1, g1, po("nop", id="nop1"))) %>>% po("featureunion", id="f1") 

level_1= level_0  %>>% gunion(list(m1, po("nop", id= "nop2")) )%>>% po("featureunion", id="f2")

level_2= level_1 %>>% gunion(list(c1, po("nop", id= "nop3"))) %>>% po("featureunion", id="f3")

level_3= level_2 %>>% po(lear4)
stack1= GraphLearner$new(level_3)
resampling= rsmp("cv", folds=3)
rr= resample(insure, stack1, resampling)



_________ python____



