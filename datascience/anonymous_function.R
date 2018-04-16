# A good rule of thumb is that an anonymous function should fit on one line and shouldn't need to use {}.
lapply(mtcars, function(x) sd(x) / mean(x))
