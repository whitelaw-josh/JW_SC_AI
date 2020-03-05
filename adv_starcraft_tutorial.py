import cv2
import keras
import math
import numpy as np
import os
import random
import time
import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STARGATE, ROBOTICSFACILITY, STALKER, VOIDRAY, OBSERVER

#os.environ["SC2PATH"] = 'C:\Program Files (x86)\StarCraft II'

HEADLESS = False

#Our BotAI class
class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.ITERATIONS_PER_MINUTE = 165 #Roughly 165 iterations (estimated)
        self.MAX_WORKERS = 50 #Max workers of 50 TODO: Might need to change over time
        self.do_something_after = 0 #Wait time to make another decision (iterations)
        self.scouts_and_spots = {} #Dictionary that tracks active scouts and what they are up to {UNIT_TAG:LOCATION}
        self.train_data = []
        self.use_model = use_model
        self.title = title

        self.choices = {
            0: self.build_scout,
            1: self.build_zealot,
            2: self.build_gateway,
            3: self.build_voidray,
            4: self.build_stalker,
            5: self.build_worker,
            6: self.build_assimilator,
            7: self.build_stargate,
            8: self.build_pylon,
            9: self.defend_nexus,
            10: self.attack_known_enemy_unit,
            11: self.attack_known_enemy_structure,
            12: self.expand,
            13: self.do_nothing
        }

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("C:/Program Files (x86)/StarCraft II/logs/BasicCNN-30-epochs-0.0001-LR-4.2")

    def on_end(self, game_result):
        print('---- on_end called ----')
        print(game_result, self.use_model)

        with open("C:/Program Files (x86)/StarCraft II/logs/gameout-random-vs-medium-log.txt", "a") as f:
            if self.use_model:
                f.write("Model {} - {}\n".format(game_result, int(time.time())))
            else:
                f.write("Random {} - {}\n".format(game_result, int(time.time()))
        
        # if game_result == Result.Victory and not self.use_model:
        #     np.save("C:/Program Files (x86)/StarCraft II/train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

#********** START OF ON_STEP FUNCTIONALITY ********** 
    async def on_step(self, iteration):
        #self.iteration = iteration
        self.game_time = (self.state.game_loop/22.4) / 60
        await self.distribute_workers() #Distribute workers evenly to mineral patches (in sc2/bot_ai.py)
        await self.scout()
        await self.intel()
        await self.do_something()

    #Method for determining random movement of unit
    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y))) #2d point based on position in sc2

        return go_to

    #Method to scout out the enemy for intel by either utilizing workers or building observers
    #Also maintains self.scouts_and_spots dictionary
    async def scout(self):
        self.expand_dis_dir = {} #Maintain enemy expansion distances from enemy start location {DISTANCE:EXPANSION_LOC}

        #Inflate expand_dis_dir with expansion distances and locations
        for el in self.expansion_locations:
            self.expand_dis_dir[el.distance_to(self.enemy_start_locations[0])] = el

        #NOTE: Might not need to sort with Python 3.7
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        
        #Removing of scouts that are dead
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 5

        assign_scout = True

        #Check to see if we need to assign a probe as a scout
        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        #Then iterate over distances from closest to furthers from enemy start location. Within each distance
        if assign_scout:
            if len(self.units(unit_type).idle) > 0: #Assign scout unit that is idle and iterate over limit type of scout unit (PROBE = 1, OBSERVER = 15)
                for obs in self.units(unit_type).idle[:unit_limit]: 
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                #Grab the location of possible enemy base
                                location = self.expand_dis_dir[dist] #next(value for key, value in self.expand_dis_dir() if key == dist)
                                #List of active locations already being scouted
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                #If not being actively scouted, assign a scout to scout location
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location

                                    break
                            except Exception as e:
                                print(str(e))
                                #pass
                                
        #Keep probes moving so they don't get assigned to minerals
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    #Method to find targets for offensive units to attack 
    # def find_target(self, state):
    #     if len(self.known_enemy_units) > 0:
    #         return random.choice(self.known_enemy_units)
    #     elif len(self.known_enemy_structures) > 0:
    #         return random.choice(self.known_enemy_structures)
    #     else:
    #         return self.enemy_start_locations[0]

    #Method to gather intelligence for an attack
    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8) #Blacked out screen of the map (general 200x200)

        for unit in self.units().ready:
            cv2.circle(game_data, (int(unit.position[0]), int(unit.position[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            cv2.circle(game_data, (int(unit.position[0]), int(unit.position[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))
   
        # for enemy_unit in self.known_enemy_units:
        #     if not enemy_unit.is_structure:
        #         worker_names = ["PROBE", "SCV", "DRONE"]

        #         if enemy_unit.name in worker_names:
        #             cv2.circle(game_data, (int(enemy_unit.position[0]), int(enemy_unit.position[1])), 1, (55, 0, 155), -1)
        #         else:
        #             cv2.circle(game_data, (int(enemy_unit.position[0]), int(enemy_unit.position[1])), 3, (55, 0, 215), -1)

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            vespene_ratio = self.vespene / 1500
            population_ratio = self.supply_left / self.supply_cap
            plausible_supply = self.supply_cap / 200.0
            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)

            if mineral_ratio > 1.0:
                mineral_ratio = 1.0  
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0       
            if population_ratio > 1.0:
                population_ratio = 1.0           
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        except Exception as e:
            print(str(e))
        
        self.flipped = cv2.flip(cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY), 0) #Coordinate system needs flipped
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not HEADLESS:
            cv2.imshow(str(self.title) resized)
            cv2.waitKey(1)

    #Method to make decision for bot to execute either via model choice or random choice
    async def do_something(self):
        if self.game_time > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else
                worker_weight = 8
                zealot_weight = 3
                voidray_weight = 20
                stalker_weight = 8
                pylon_weight = 5
                stargate_weight = 5
                gateway_weight = 3

                choice_weights = 1*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+worker_weight*[5]+1*[6]+stargate_weight*[7]+pylon_weight*[8]+1*[9]+1*[10]+1*[11]+1*[12]+1*[13]
                choice = random.choice(choice_weights)
            
            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))

            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped])

#********** END OF ON_STEP FUNCTIONALITY ********** 

#********** START OF CHOICE FUNCTIONALITY ********** 
    #Method to build scouting units and robotics facility if one is not already built
    #TODO: Second check in the first if statement is to make sure we do not keep building observers past the unit limit defined in scout
    #(5 for observers).  The 6th observer would be useful to move with the army
    #TODO: Move building robotics facility into separate method
    async def build_scout(self):
        if self.units(CYBERNETICSCORE).ready.exists and len(self.units(ROBOTICSFACILITY)) < 1:
            if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                await self.build(ROBOTICSFACILITY), near=pylon)

        if len(self.units(OBSERVER)) < math.floor(self.game_time/3) and len(self.units(OBSERVER)) < 5: #Want to build observers every 3 minutes
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                #print(len(self.units(OBSERVER)), self.game_time/3)
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
    
    #Method to build zealot units
    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready

        if gateways.exists and self.can_afford(ZEALOT):
            await self.do(random.choice(gateways).train(ZEALOT))

    #Method to build gateway buildings
    #TODO: Time check to build gateways (Equation to calculate number of gateways to build at certain times)
    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random

        if pylon.exists and self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    #Method to build voidray units
    async def build_voidray(self):
        stargates = self.units(STARGATE).ready

        if stargates.exists and self.can_afford(VOIDRAY):
            await self.do(random.choice(stargates).train(VOIDRAY))
    
    #Method to build stalker units
    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if not cybernetics_cores.exists and self.unit(GATEWAY).ready.exists:
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=pylon)

        if gateways.exists and cybernetics_cores.exists and self.can_afford(STALKER):
            await self.do(random.choice(gateways).train(STALKER))

    #Method to build workers at a nexus if affordable
    #TODO: Update method to take into account # of nexuses with # of workers with regard to game_time as well and not go past MAX_WORKERS 
    async def build_worker(self):
        nexuses = self.units(NEXUS).ready

        if nexuses.exists and self.can_afford(PROBE):
            await self.do(random.choice(nexuses).train(PROBE))

    #Method to build assimilator within 15 units of a nexus and if affordable
    #Find worker that is available and geyser that is available to have assimilator built
    #TODO: May need to update location of building assimilators
    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            geysers = self.state.vespene_geyser.closer_than(15.0, nexus)
            
            for gas in geysers:
                if not self.can_afford(ASSIMILATOR):
                    break

                worker = self.select_build_worker(gas.position)

                if worker is None:
                    break

                if not self.units(ASSIMILATOR).closer_than(1.0, gas).exists:
                    await self.do(worker.build(ASSIMILATOR, gas))

    #Method to build stargate building
    async def build_stargate(self):
        pylon = self.units(PYLON).ready.random

        if pylon.exists and self.units(CYBERNETICSCORE).ready.exists:
            if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                await self.build(STARGATE, near=pylon)

    #Method to build pylon if supply left is less than 5 and pylon is not already being built
    #Will build pylon near nexus first and if affordable
    #TODO: May need to update location of building pylons
    async def build_pylon(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready

            if nexuses.exists and self.can_afford(PYLON):
                await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    #Method to assign units to target
    #TODO: Might be better to get a list of attack units and apply target for attacking units across said list (i.e. one liner?)
    def attack_target(self, target):
        for u in self.units(VOIDRAY).idle:
            await self.do(u.attack(target))
        for u in self.units(STALKER).idle:
            await self.do(u.attack(target))
        for u in self.units(ZEALOT).idle:
            await self.do(u.attack(target))

    #Method to defend nexus from enemy units
    async def defend_nexus(self):
        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
        self.attack_target(target)

    #Method to attack enemy units if known
    async def attack_known_enemy_unit(self):
        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
        self.attack_target(target)

    async def attack_known_enemy_structure(self):
        target = random.choice(self.known_enemy_structures)
        self.attack_target(target)

    #Method to build nexus if less than 2 nexus on the map and can afford it
    async def expand(self):
        try:
            if self.units(NEXUS).amount < self.game_time / 2 and self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    #Method to do_nothing and wait
    async def do_nothing(self):
        self.do_something_after = self.game_time + (random.randrange(7, 100)/100)

#********** END OF CHOICE FUNCTIONALITY ********** 

#Starts the game we want to run taking into the map and players we wish to add
#realtime controls speed of how game is played
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime = False)