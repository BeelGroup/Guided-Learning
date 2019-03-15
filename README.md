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