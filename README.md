We present the the concept of Guided Learning, which out-lines a framework in which a reinforcement learning agent can effectively’ask for help’ as it encounters stagnation. Either a human or expert agentsupervisor can then effectively ’guide’ the agent as to how to progressbeyond the point of stagnation. This guidance is then encoded in a novelway using a separately trained neural network referred to as a ’TaughtResponse  Memory’  that  can  be  recalled  when  another  ’similar’  situa-tion  arises  in  the  future.  This  paper  applies  Guided  Learning  on  topof  an  evolutionary  algorithm  but  also  shows  how  Guided  Learning  isalgorithm independent and can be applied in any reinforcement learn-ing context. The results show that our initial implementation of GuidedLearning provided in this paper gives superior performance and yields,on average, an increase of 136% in the rate of progression of the mostfit  genome  with  best  and  worst  case  results  yielding  137%  and  110%respectively and an average increase of 112% in rate of progression forthe average genome with best and worst case results of 558% and 47%respectively. All results were achieved with minimal guidance. Such re-sults  occur  because  the  agent  can  exploit  more  information  and  thus,the need for exploration of the solution space is reduced. The results ob-tained show good promise for Guided Learnings potential as such resultswere obtained with only a partial implementation and much future workstill remains.

## Setup
Dependancies:
* Python 3.6 + various libraries (matplotlib, sklearn, skimage, etc..)
* OpenAI Retro: https://github.com/openai/retro

python -m pip install -r requirements.txt

python -m retro.import /path/to/your/ROMs/directory/

copy data.json to \path_to_python\Lib\site-packages\retro\data\stable\SuperMarioBros-Nes

Patch python-neat by adding:
```
    def next_generation(self):
        # Gather and report statistics.
        best = None
        for g in itervalues(self.population):
            if best is None or g.fitness > best.fitness:
                best = g
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        self.reporters.end_generation(self.config, self.population, self.species)

        self.generation += 1

        return
```

To the end of \path_to_python\Lib\site-packages\neat\population.py
