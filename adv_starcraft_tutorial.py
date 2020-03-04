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
    def __init__(self, use_model=True):
        self.ITERATIONS_PER_MINUTE = 165 #Roughly 165 iterations (estimated)
        self.MAX_WORKERS = 50 #Max workers of 50
        self.do_something_after = 0 #Wait time to make another decision (iterations)
        self.scouts_and_spots = {} #Dictionary that tracks active scouts and what they are up to {UNIT_TAG:LOCATION}
        self.train_data = []
        self.use_model = use_model

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("C:/Program Files (x86)/StarCraft II/logs/BasicCNN-30-epochs-0.0001-LR-4.2")

    def on_end(self, game_result):
        print('---- on_end called ----')
        print(game_result, self.use_model)

        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))
        
        if game_result == Result.Victory and not self.use_model:
            np.save("C:/Program Files (x86)/StarCraft II/train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):
        #self.iteration = iteration
        self.game_time = (self.state.game_loop/22.4) / 60
        await self.build_scout()
        await self.scout()
        await self.distribute_workers() #Distribute workers evenly to mineral patches (in sc2/bot_ai.py)
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

    #Method for determining random movement of unit
    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

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

    #Method to build scouting units
    #TODO: Second check in the first if statement is to make sure we do not keep building observers past the unit limit defined in scout
    #(5 for observers).  The 6th observer would be useful to move with the army
    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.game_time/3) and len(self.units(OBSERVER)) < 5: #Want to build observers every 3 minutes
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                print(len(self.units(OBSERVER)), self.game_time/3)
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

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


    #Method to build workers at NEXUS if affordable
    #TODO: consider if 16 should be switched to 22 for workers in mineral patches and assimilators
    async def build_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    #Method to build pylon if supply left is less than 5 and pylon is not already being built
    #Will build pylon near nexus first and if afforadble
    #TODO: May need to update location of building pylons
    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexi = self.units(NEXUS).ready
            if nexi.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexi.first)

    #Method to build assimilator within 25 units of a nexus and if affordable
    #Find worker that is available and geyser that is available to have assimilator built
    #TODO: May need to update location of building assimilators
    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            geysers = self.state.vespene_geyser.closer_than(25.0, nexus)
            
            for gas in geysers:
                if not self.can_afford(ASSIMILATOR):
                    break

                worker = self.select_build_worker(gas.position)

                if worker is None:
                    break

                if not self.units(ASSIMILATOR).closer_than(1.0, gas).exists:
                    await self.do(worker.build(ASSIMILATOR, gas))

    #Method to build nexus if less than 2 nexi on the map and can afford it
    async def expand(self):
        try: 
            if self.units(NEXUS).amount < self.game_time / 2 and self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    #Method to build offensive buildings (i.e. Race = Protoss, Stargate, RoboticsFacility)
    #TODO: May need to update location of building offensive force buildings
    async def build_offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            
            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
                    
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

                if len(self.units(STARGATE)) < self.game_time:
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    #Method to build offensive units (i.e.  Race = Protoss, Voidray ONLY)
    async def build_offensive_force(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    #Method to gather intelligence for an attack
    async def intel(self):
        draw_dict = { #Dictonary of colors for specific structures, units, 
                NEXUS: [15, (0, 255, 0)],
                PYLON: [3, (20, 235, 0)],
                PROBE: [1, (55, 200, 0)],
                ASSIMILATOR: [2, (55, 200, 0)],
                GATEWAY: [3, (200, 100, 0)],
                CYBERNETICSCORE: [3, (150, 150, 0)],
                STARGATE: [5, (255, 0, 0)],
                VOIDRAY: [3, (255, 100, 0)],
                ROBOTICSFACILITY: [5, (215, 155, 0)],
            }

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8) #Blacked out screen of the map (general 200x200)

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                #print(unit.position)
                cv2.circle(game_data, (int(unit.position[0]), int(unit.position[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1) #Draw a circle on the unit position

        for obs in self.units(OBSERVER).ready:
            cv2.circle(game_data, (int(obs.position[0]), int(obs.position[1])), 1, (255, 255, 255), -1)
   
        line_max = 50
        mineral_ratio = self.minerals / 1500
        vespene_ratio = self.vespene / 1500
        population_ratio = self.supply_left / self.supply_cap
        plausible_supply = self.supply_cap / 200.0
        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left) #TODO: Update this based on different army compositions

        if mineral_ratio > 1.0:
            mineral_ratio = 1.0           
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0       
        if population_ratio > 1.0:
            population_ratio = 1.0           
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        main_base_names = ["NEXUS", "COMMANDERCENTER", "HATCHERY"]

        for enemy_building in self.known_enemy_structures:
            if enemy_building.name not in main_base_names:
                cv2.circle(game_data, (int(enemy_building.position[0]), int(enemy_building.position[1])), 5, (200, 50, 212), -1)
            else:
                cv2.circle(game_data, (int(enemy_building.position[0]), int(enemy_building.position[1])), 15, (0, 0, 255), -1)
        
        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["PROBE", "SCV", "DRONE"]

                if enemy_unit.name in worker_names:
                    cv2.circle(game_data, (int(enemy_unit.position[0]), int(enemy_unit.position[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(enemy_unit.position[0]), int(enemy_unit.position[1])), 3, (55, 0, 215), -1)
        
        self.flipped = cv2.flip(game_data, 0) #Coordinate system needs flipped

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    #Method to find targets for offensive units to attack 
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    #Method to make offensive units attack and defend
    #TODO: Conditional to check if being attacked (i.e. defend)
    #TODO: Might need to have variable (local or outside of scope of attack method) containing sum of all offensive units
    async def attack(self):
        if len(self.units(VOIDRAY).idle) > 0:   
            target = False

            if self.game_time > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])
                    #print('prediction: ',choice)

                    choice_dict = {
                        0: "No attack",
                        1: "Attack close to our nexus",
                        2: "Attack enemy structure",
                        3: "Attack enemy start"
                    }

                    print("Choice #{}:{}".format(choice, choice_dict[choice]))
                else:
                    choice = random.randrange(0, 4)

                if choice == 0: #Do not attack
                    wait = random.randrange(7,100)/100
                    self.do_something_after = self.game_time + wait
                elif choice == 1: #Attack units closest our nexus (random)
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS))) #TODO: Update choice of nexus
                elif choice == 2: #Attack enemy structures if known
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                elif choice == 3: #Attack enemy start location
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))
                
                y = np.zeros(4)
                y[choice] = 1
                #print(y)
                self.train_data.append([y, self.flipped])

#Starts the game we want to run taking into the map and players we wish to add
#realtime controls speed of how game is played
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime = False)