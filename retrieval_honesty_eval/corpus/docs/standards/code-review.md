# Code review standard

This standard is binding for every change merged into Lantern.

A change may be merged only when all three conditions hold: at least one reviewer other than the author has approved it; every test in the continuous integration run has passed; and any change to a database schema ships with a reversible migration.

Reviewers should prefer small changes, but size alone is never a reason to block a merge.
