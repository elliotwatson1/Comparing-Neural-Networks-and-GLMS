# Load libraries
library(sf)
library(ggplot2)
library(dplyr)
library(geodata)
library(rnaturalearth)
library(rnaturalearthdata)

####################################################################################################################
# Count map data

# --- 1. Download Sweden counties (GADM level 1) ---
sweden_counties <- gadm(country = "SWE", level = 1, path = tempdir())
sweden_counties_sf <- st_as_sf(sweden_counties)

# --- 2. Create a county-to-zone mapping ---
county_zone_map <- tibble(
  NAME_1 = c("Stockholm", "Västra Götaland", "Skåne", "Uppsala", "Södermanland", "Östergötland",
             "Jönköping", "Kronoberg", "Kalmar", "Gotland", "Blekinge", "Halland", "Värmland",
             "Örebro", "Västmanland", "Dalarna", "Gävleborg", "Västernorrland", "Jämtland",
             "Västerbotten", "Norrbotten", "Orebro"),
  Zone = c(1,1,1,2,2,2,2,3,3,7,3,3,4,4,4,4,4,5,5,6,6, 2)
)

# Merge zone info into county shapefile
sweden_counties_sf <- left_join(sweden_counties_sf, county_zone_map, by = "NAME_1")

# --- 3. Aggregate Ohlsson data by zone ---
zone_counts <- dataOhlsson %>%
  group_by(zon) %>%
  summarise(count = n(), .groups = "drop")

# --- 4. Merge Ohlsson counts with county shapefile ---
# Summarise counts per county by matching zon to county Zone
county_counts <- sweden_counties_sf %>%
  left_join(zone_counts, by = c("Zone" = "zon"))

# Replace NA counts with 0
county_counts$count[is.na(county_counts$count)] <- 0

# --- 5. Optional: add major cities ---
cities <- ne_download(scale = "medium", type = "populated_places", category = "cultural", returnclass = "sf")
sweden <- ne_countries(scale = "medium", country = "Sweden", returnclass = "sf")
sweden_cities <- st_intersection(cities, sweden)  # keep only cities in Sweden

# --- 6. Plot choropleth map ---
ggplot() +
  geom_sf(data = county_counts, aes(fill = count), color = "black") +
  scale_fill_viridis_c(option = "plasma", name = "Count") +
  geom_sf(data = sweden_cities, color = "red", size = 3) +
  labs(title = "Ohlsson Insurance Data by Swedish Zone",
       subtitle = "Red dots indicate major cities") +
  theme_minimal()
# Number of counts by zone

####################################################################################################################


# Add this to appendix to further see the zones when describing in the EDA
# Plot
ggplot(data = sweden_counties_sf) +
  geom_sf(aes(fill = factor(Zone)), color = "black") +
  scale_fill_viridis_d(option = "turbo", name = "Zone") +
  labs(title = "Swedish Counties by Zone") +
  theme_minimal()

####################################################################################################################
# Claims per zone
zone_counts <- dataOhlsson %>%
  group_by(zon) %>%
  summarise(total_claims = sum(antskad), .groups = "drop")

county_counts <- sweden_counties_sf %>%
  left_join(zone_counts, by = c("Zone" = "zon"))

ggplot(county_counts) +
  geom_sf(aes(fill = total_claims), color = "black") +
  scale_fill_viridis_c(option = "rocket") +
  geom_sf(data = sweden_cities, color = "blue", size = 6) +
  labs(title = "Total Claims per Zone") +
  theme_minimal()

####################################################################################################################
# Average Claim cost per zone
zone_avg_cost <- dataOhlsson %>%
  group_by(zon) %>%
  summarise(avg_cost = mean(skadkost[antskad > 0], na.rm = TRUE), .groups = "drop")

county_cost <- sweden_counties_sf %>%
  left_join(zone_avg_cost, by = c("Zone" = "zon"))

ggplot(county_cost) +
  geom_sf(aes(fill = avg_cost), color = "black") +
  scale_fill_viridis_c(option = "magma") +
  labs(title = "Average Claim Cost per Zone") +
  theme_minimal()

####################################################################################################################
# Claim frequency by zone

zone_freq <- dataOhlsson %>%
  group_by(zon) %>%
  summarise(freq = sum(antskad) / sum(duration), .groups = "drop")

county_freq <- sweden_counties_sf %>%
  left_join(zone_freq, by = c("Zone" = "zon"))

ggplot(county_freq) +
  geom_sf(aes(fill = freq), color = "black") +
  scale_fill_viridis_c(option = "viridis") +
  labs(title = "Claim Frequency per Zone") +
  theme_minimal()

####################################################################################################################
# Facet map by motorcycle class - unsure what this is double check
zone_mcklass <- dataOhlsson %>%
  group_by(zon, mcklass) %>%
  summarise(freq = sum(antskad) / sum(duration), .groups = "drop")

county_mcklass <- sweden_counties_sf %>%
  left_join(zone_mcklass, by = c("Zone" = "zon"))

ggplot(county_mcklass) +
  geom_sf(aes(fill = freq), color = "black") +
  facet_wrap(~mcklass) +
  scale_fill_viridis_c(option = "cividis") +
  labs(title = "Claim Frequency by Motorcycle Class") +
  theme_minimal()

####################################################################################################################
# Bubble map of average claim. size
zone_avg_cost <- dataOhlsson %>%
  group_by(zon) %>%
  summarise(avg_cost = mean(skadkost[antskad > 0], na.rm = TRUE), .groups = "drop")

zone_centroids <- sweden_counties_sf %>%
  group_by(Zone) %>%
  summarise() %>%
  st_centroid()

bubble_data <- left_join(zone_centroids, zone_avg_cost, by = c("Zone" = "zon"))

ggplot() +
  geom_sf(data = sweden_counties_sf, fill = "white", color = "black") +
  geom_sf(data = bubble_data, aes(size = avg_cost), color = "red", alpha = 0.6) +
  scale_size_continuous(name = "Avg Claim Cost") +
  labs(title = "Average Claim Cost (Bubble Map)") +
  theme_minimal()

####################################################################################################################

