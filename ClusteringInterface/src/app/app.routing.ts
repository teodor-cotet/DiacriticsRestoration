import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router'
import { ClusteringComponent } from './clustering/clustering.component';

const appRoutes: Routes = [
    {
        path: 'clustering',
        component: ClusteringComponent
    }
];

@NgModule({
    imports: [
        RouterModule.forRoot(appRoutes)
    ],
    exports: [
        RouterModule
    ],
    declarations: []
})

export class AppRoutingModule {
}
