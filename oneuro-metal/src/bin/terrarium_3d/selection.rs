//! Entity selection state for click-to-select and Tab cycling.

use super::mesh::EntityTag;

pub struct Selection {
    pub tag: EntityTag,
    pub cycle_list: Vec<EntityTag>,
    pub cycle_idx: usize,
}

impl Selection {
    pub fn new() -> Self {
        Self { tag: EntityTag::None, cycle_list: Vec::new(), cycle_idx: 0 }
    }

    pub fn select(&mut self, tag: EntityTag) {
        self.tag = tag;
        // Sync cycle index to match selection
        if let Some(pos) = self.cycle_list.iter().position(|t| *t == tag) {
            self.cycle_idx = pos;
        }
    }

    pub fn deselect(&mut self) {
        self.tag = EntityTag::None;
    }

    pub fn cycle_next(&mut self) {
        if self.cycle_list.is_empty() { return; }
        self.cycle_idx = (self.cycle_idx + 1) % self.cycle_list.len();
        self.tag = self.cycle_list[self.cycle_idx];
    }

    pub fn update_cycle_list(&mut self, plants: usize, flies: usize, waters: usize, fruits: usize) {
        self.cycle_list.clear();
        for i in 0..plants { self.cycle_list.push(EntityTag::Plant(i)); }
        for i in 0..flies { self.cycle_list.push(EntityTag::Fly(i)); }
        for i in 0..waters { self.cycle_list.push(EntityTag::Water(i)); }
        for i in 0..fruits { self.cycle_list.push(EntityTag::Fruit(i)); }
    }

    pub fn is_selected(&self) -> bool {
        self.tag != EntityTag::None
    }

    pub fn label(&self) -> String {
        match self.tag {
            EntityTag::None => "none".into(),
            EntityTag::Terrain => "terrain".into(),
            EntityTag::Plant(i) => format!("Plant #{}", i),
            EntityTag::Fly(i) => format!("Fly #{}", i),
            EntityTag::Water(i) => format!("Water #{}", i),
            EntityTag::Fruit(i) => format!("Fruit #{}", i),
            EntityTag::Atom(i) => format!("Atom #{}", i),
            EntityTag::Bond(i) => format!("Bond #{}", i),
            EntityTag::Metabolite(i) => format!("Metabolite #{}", i),
        }
    }
}
